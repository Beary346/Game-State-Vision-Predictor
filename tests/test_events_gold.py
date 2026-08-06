"""Tests for the Gold event layer (state-delta aggregation).

AGENTS.md contract: changes between consecutive Silver state tuples become
events --

- Health drop (enemy)          -> ``hit_landed``   (a hit was landed)
- Health drop (player)         -> ``hit_taken``    (the player got hit)
- No return hit after landing  -> ``punish``       (free damage, no reply)
- Attack without a landing     -> ``whiff``
- Death (player / enemy)       -> ``round_loss`` / ``round_win``
- Domain alert (rising edge)   -> ``domain_alert`` (domain deployed)
"""

import pytest

from src.pipeline.events_gold import EVENT_TYPES, Event, detect_events


def _state(
    frame_index: int = 0,
    player_health: float = 1.0,
    enemy_healths: list[float] | None = None,
    attacking: bool = False,
    defending: bool = False,
    domain_ready: bool = False,
    round_index: int = 0,
    timestamp_sec: float | None = None,
) -> dict:
    """Build one SilverFeatures-style state dict for event tests."""
    enemy_healths = enemy_healths if enemy_healths is not None else [0.8]
    return {
        "image_path": "/fake/frame.png",
        "player_health": player_health,
        "player_position": [0.5, 0.5],
        "enemies": [
            {
                "bbox": [100, 200, 60, 60],
                "health": h,
                "health_bar_bbox": [100, 170, 50, 8],
                "confidence": 0.9,
            }
            for h in enemy_healths
        ],
        "attacking": attacking,
        "defending": defending,
        "damage_indicator": False,
        "num_enemies": len(enemy_healths),
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec if timestamp_sec is not None else float(frame_index) / 30.0,
        "clock_sec": 60.0 - frame_index,
        "domain_ready": domain_ready,
        "ocr_confidence": 0.95,
        "round_index": round_index,
    }


def _types(events: list[Event]) -> list[str]:
    """The event types in frame order."""
    return [e.type for e in events]


# ── Hit events (health drops) ────────────────────────────────────────────────


class TestHealthDropEvents:
    def test_enemy_health_drop_is_hit_landed(self):
        states = [
            _state(0, enemy_healths=[0.8]),
            _state(1, enemy_healths=[0.5]),
            _state(2, enemy_healths=[0.5]),
        ]
        events = detect_events(states, punish_window_frames=0)
        assert _types(events) == ["hit_landed"]
        hit = events[0]
        assert hit.frame_index == 1
        assert hit.detail["health_before"] == pytest.approx(0.8)
        assert hit.detail["health_after"] == pytest.approx(0.5)

    def test_player_health_drop_is_hit_taken(self):
        states = [
            _state(0, player_health=1.0),
            _state(1, player_health=0.7),
            _state(2, player_health=0.7),
        ]
        events = detect_events(states)
        assert _types(events) == ["hit_taken"]

    def test_subthreshold_drop_produces_no_event(self):
        states = [
            _state(0, player_health=1.0, enemy_healths=[0.8]),
            _state(1, player_health=0.97, enemy_healths=[0.79]),
        ]
        assert detect_events(states) == []

    def test_health_increase_is_not_a_hit(self):
        states = [
            _state(0, player_health=0.4),
            _state(1, player_health=0.9),
        ]
        assert detect_events(states) == []

    def test_mean_enemy_health_used_for_multiple_enemies(self):
        states = [
            _state(0, enemy_healths=[0.8, 0.8]),
            _state(1, enemy_healths=[0.6, 0.8]),
            _state(2, enemy_healths=[0.6, 0.8]),
        ]
        events = detect_events(states, punish_window_frames=0)
        # mean 0.8 -> 0.7 = 0.1 drop, above the 0.06 threshold.
        assert _types(events) == ["hit_landed"]

    def test_damage_recorded_in_hit_detail(self):
        states = [_state(0, player_health=0.9), _state(1, player_health=0.55)]
        events = detect_events(states)
        assert events[0].type == "hit_taken"
        assert events[0].detail["damage"] == pytest.approx(0.35)


# ── Punish: no return hit ────────────────────────────────────────────────────


class TestPunish:
    def test_hit_without_return_is_punish(self):
        # Player lands a hit (enemy health drops) and is never hit back.
        states = [
            _state(0, enemy_healths=[0.8]),
            _state(1, enemy_healths=[0.5]),
            _state(2, enemy_healths=[0.5]),
            _state(3, enemy_healths=[0.5]),
        ]
        events = detect_events(states)
        assert _types(events) == ["hit_landed", "punish"]
        assert events[1].frame_index == 1

    def test_hit_with_return_is_not_punish(self):
        # Enemy returns damage within the punish window (even on the same
        # frame-pair -- that is a trade, not free damage).
        states = [
            _state(0, player_health=1.0, enemy_healths=[0.8]),
            _state(1, player_health=0.8, enemy_healths=[0.5]),
            _state(2, player_health=0.8, enemy_healths=[0.5]),
        ]
        events = detect_events(states)
        assert "punish" not in _types(events)
        assert _types(events) == ["hit_landed", "hit_taken"]

    def test_return_one_frame_after_is_not_punish(self):
        states = [
            _state(0, player_health=1.0, enemy_healths=[0.8]),
            _state(1, player_health=1.0, enemy_healths=[0.5]),
            _state(2, player_health=0.7, enemy_healths=[0.5]),
        ]
        events = detect_events(states)
        assert "punish" not in _types(events)

    def test_return_outside_window_is_still_punish(self):
        # The return hit lands after punish_window_frames frames.
        states = [
            _state(0, player_health=1.0, enemy_healths=[0.8]),
            _state(1, player_health=1.0, enemy_healths=[0.5]),
            _state(2, player_health=1.0, enemy_healths=[0.5]),
            _state(3, player_health=1.0, enemy_healths=[0.5]),
            _state(4, player_health=1.0, enemy_healths=[0.5]),
            _state(5, player_health=0.7, enemy_healths=[0.5]),
        ]
        events = detect_events(states, punish_window_frames=3)
        assert "punish" in _types(events)

    def test_punish_window_parameter_respected(self):
        # The return arrives 5 frames after the landing: inside a wide window
        # it is a reply, outside it (or disabled) it is a punish.
        states = [
            _state(0, player_health=1.0, enemy_healths=[0.8]),
            _state(1, player_health=1.0, enemy_healths=[0.5]),
            _state(2, player_health=1.0, enemy_healths=[0.5]),
            _state(3, player_health=1.0, enemy_healths=[0.5]),
            _state(4, player_health=1.0, enemy_healths=[0.5]),
            _state(5, player_health=1.0, enemy_healths=[0.5]),
            _state(6, player_health=0.7, enemy_healths=[0.5]),
        ]
        assert "punish" in _types(detect_events(states, punish_window_frames=3))
        assert "punish" not in _types(detect_events(states, punish_window_frames=6))
        assert "punish" not in _types(detect_events(states, punish_window_frames=0))


# ── Whiff: attack with no landing ────────────────────────────────────────────


class TestWhiff:
    def test_attack_without_landing_is_whiff(self):
        states = [
            _state(0, attacking=False, enemy_healths=[0.8]),
            _state(1, attacking=True, enemy_healths=[0.8]),
            _state(2, attacking=True, enemy_healths=[0.8]),
            _state(3, attacking=False, enemy_healths=[0.8]),
        ]
        events = detect_events(states)
        assert "whiff" in _types(events)
        whiff = next(e for e in events if e.type == "whiff")
        assert whiff.frame_index == 1

    def test_attack_that_lands_is_not_whiff(self):
        states = [
            _state(0, attacking=False, enemy_healths=[0.8]),
            _state(1, attacking=True, enemy_healths=[0.5]),
        ]
        assert "whiff" not in _types(detect_events(states))

    def test_whiff_window_parameter_respected(self):
        # The landing arrives two frames after the onset: inside a 3-frame
        # window it is a hit, outside it is a whiff.
        states = [
            _state(0, attacking=False, enemy_healths=[0.8]),
            _state(1, attacking=True, enemy_healths=[0.8]),
            _state(2, attacking=True, enemy_healths=[0.8]),
            _state(3, attacking=True, enemy_healths=[0.5]),
        ]
        assert "whiff" not in _types(detect_events(states, whiff_window_frames=3))
        assert "whiff" in _types(detect_events(states, whiff_window_frames=1))

    def test_window_zero_disables_whiff_detection(self):
        states = [
            _state(0, attacking=False),
            _state(1, attacking=True),
            _state(2, attacking=False),
        ]
        assert detect_events(states, whiff_window_frames=0) == []

    def test_sustained_attack_only_whiffs_once(self):
        states = [
            _state(0, attacking=False),
            _state(1, attacking=True),
            _state(2, attacking=True),
            _state(3, attacking=True),
        ]
        events = detect_events(states)
        assert _types(events).count("whiff") == 1


# ── Death / round outcome ────────────────────────────────────────────────────


class TestDeathEvents:
    def test_player_death_is_round_loss(self):
        # A 0.05 drop stays below the hit threshold, so only the death fires.
        states = [
            _state(0, player_health=0.1),
            _state(1, player_health=0.05),
        ]
        events = detect_events(states)
        assert _types(events) == ["round_loss"]

    def test_enemy_death_is_round_win(self):
        states = [
            _state(0, enemy_healths=[0.09]),
            _state(1, enemy_healths=[0.04]),
        ]
        events = detect_events(states)
        assert _types(events) == ["round_win"]

    def test_death_event_fires_once(self):
        # Health stays at zero across many frames: one event only.
        states = [_state(0, player_health=0.2)]
        states += [_state(i, player_health=0.0) for i in range(1, 6)]
        events = detect_events(states)
        assert _types(events).count("round_loss") == 1

    def test_recovery_resets_death_flag(self):
        states = [
            _state(0, player_health=0.2),
            _state(1, player_health=0.0),   # round loss
            _state(2, player_health=1.0),   # new round, full health
            _state(3, player_health=0.6),   # normal play
            _state(4, player_health=0.0),   # second round loss
        ]
        events = detect_events(states)
        assert _types(events).count("round_loss") == 2


# ── Domain alert ─────────────────────────────────────────────────────────────


class TestDomainAlert:
    def test_domain_ready_rising_edge_alerts(self):
        states = [
            _state(0, domain_ready=False),
            _state(1, domain_ready=True),
            _state(2, domain_ready=True),
        ]
        events = detect_events(states)
        assert _types(events) == ["domain_alert"]
        assert events[0].frame_index == 1

    def test_no_alert_when_domain_stays_ready(self):
        states = [
            _state(0, domain_ready=True),
            _state(1, domain_ready=True),
        ]
        assert detect_events(states) == []

    def test_alert_again_after_dropping(self):
        states = [
            _state(0, domain_ready=True),
            _state(1, domain_ready=False),
            _state(2, domain_ready=True),
        ]
        events = detect_events(states)
        assert _types(events).count("domain_alert") == 1


# ── Round boundaries ─────────────────────────────────────────────────────────


class TestRoundBoundaries:
    def test_no_spurious_hits_across_round_reset(self):
        # Frame 1 -> 2 is a round transition: player health jumps back up
        # (1.0 -> 0.2 -> 1.0) and must not look like a hit landed / healed.
        states = [
            _state(0, player_health=1.0, enemy_healths=[0.8], round_index=0),
            _state(1, player_health=0.2, enemy_healths=[0.8], round_index=0),
            _state(2, player_health=1.0, enemy_healths=[0.8], round_index=1),
            _state(3, player_health=1.0, enemy_healths=[0.8], round_index=1),
        ]
        events = detect_events(states)
        assert _types(events) == ["hit_taken"]

    def test_round_index_and_timestamps_attached(self):
        states = [
            _state(0, player_health=1.0, round_index=0),
            _state(1, player_health=0.6, round_index=0),
        ]
        event = detect_events(states)[0]
        assert event.round_index == 0
        assert event.timestamp_sec == pytest.approx(1.0 / 30.0)

    def test_events_sorted_by_frame_index(self):
        states = [
            _state(0, player_health=1.0, enemy_healths=[0.8], attacking=False, domain_ready=False),
            _state(1, player_health=0.6, enemy_healths=[0.8], attacking=True, domain_ready=True),
            _state(2, player_health=0.6, enemy_healths=[0.4], attacking=True, domain_ready=True),
        ]
        events = detect_events(states)
        frames = [e.frame_index for e in events]
        assert frames == sorted(frames)
        assert _types(events) == ["hit_taken", "domain_alert", "hit_landed", "punish"]


# ── Input contract ───────────────────────────────────────────────────────────


class TestInputContract:
    def test_accepts_dataclass_objects(self):
        from src.pipeline.silver import SilverFeatures

        states = [
            SilverFeatures(player_health=1.0, enemies=[{"health": 0.8, "bbox": [1, 1, 2, 2], "health_bar_bbox": [1, 1, 2, 2], "confidence": 1.0}], num_enemies=1),
            SilverFeatures(player_health=1.0, enemies=[{"health": 0.5, "bbox": [1, 1, 2, 2], "health_bar_bbox": [1, 1, 2, 2], "confidence": 1.0}], num_enemies=1),
        ]
        events = detect_events(states, punish_window_frames=0)
        assert _types(events) == ["hit_landed"]

    def test_empty_stream_yields_no_events(self):
        assert detect_events([]) == []

    def test_single_frame_yields_no_events(self):
        assert detect_events([_state(0)]) == []

    def test_event_types_are_known(self):
        events = detect_events(
            [
                _state(0, player_health=1.0, enemy_healths=[0.8], attacking=False, domain_ready=False),
                _state(1, player_health=0.0, enemy_healths=[0.0], attacking=True, domain_ready=True),
            ]
        )
        for event in events:
            assert event.type in EVENT_TYPES
            assert isinstance(event, Event)
            assert event.detail is not None
