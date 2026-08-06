"""Gold event layer: state-delta aggregation into match events.

AGENTS.md defines this layer as the bridge between per-frame Silver state
tuples and the match report. Changes between consecutive state tuples become
events:

- Enemy health drop                     -> ``hit_landed``  (a hit was landed)
- Player health drop                    -> ``hit_taken``   (the player got hit)
- A hit landed with no return hit       -> ``punish``       (free damage, no reply)
- An attack with no landing             -> ``whiff``
- Player health hits zero               -> ``round_loss``
- Enemy health hits zero                -> ``round_win``
- Domain readiness rising edge          -> ``domain_alert`` (domain deployed)

Every event carries the frame index, timestamp, round index, and a detail dict
so the report layer can aggregate damage, round-by-round counts, and the score.
"""

from dataclasses import dataclass, field

# Every event type the report layer understands. Adding a type here and to
# detect_events() keeps the headline/stats and the timeline in sync.
EVENT_TYPES: tuple[str, ...] = (
    "hit_landed",
    "hit_taken",
    "punish",
    "whiff",
    "round_loss",
    "round_win",
    "domain_alert",
)


@dataclass
class Event:
    """One state-delta event.

    ``detail`` carries event-specific values, e.g. for hits the health snapshot
    (health_before / health_after / damage) that the report uses to sum
    damage per round.
    """

    type: str
    frame_index: int
    timestamp_sec: float
    round_index: int
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serializable form used by the timeline JSON."""
        return {
            "type": self.type,
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "round_index": self.round_index,
            "detail": self.detail,
        }


def _state_field(state, key: str, default):
    """Read *key* from a SilverFeatures dict or dataclass alike."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _mean_enemy_health(state) -> float:
    """Mean health over all detected enemies on a frame (0.0 if none)."""
    enemies = _state_field(state, "enemies", None) or []
    healths = [
        float(e) if isinstance(e, (int, float)) else float(e.get("health", 0.0))
        for e in enemies
    ]
    return float(sum(healths) / len(healths)) if healths else 0.0


def detect_events(
    states: list,
    *,
    health_drop_threshold: float = 0.06,
    death_threshold: float = 0.05,
    punish_window_frames: int = 5,
    whiff_window_frames: int = 4,
) -> list[Event]:
    """Turn a stream of Silver state tuples into a sorted list of events.

    Health comparisons only run within a round: a round boundary (changed
    ``round_index`` or a > 0.5 health recovery) resets the baseline so the
    health reset itself is never mistaken for a hit. Death events fire once
    per continuous death episode, and recovery re-arms the detector.

    Parameters
    ----------
    states
        Iterable of per-frame state tuples -- SilverFeatures instances or the
        dicts ``asdict`` produces. Keys used: ``player_health``, ``enemies``,
        ``attacking``, ``domain_ready``, ``round_index``, ``frame_index``,
        ``timestamp_sec``.
    health_drop_threshold : float
        Minimum drop (0..1) between frames to register a hit event.
    death_threshold : float
        Health at or below this counts as dead (round outcome).
    punish_window_frames : int
        How many frames a ``hit_landed`` stays "returnable" after it lands
        (a return on the same frame pair counts as a trade, not a punish).
        A landing with no ``hit_taken`` inside the window is a punish;
        ``<= 0`` disables punish detection.
    whiff_window_frames : int
        How many frames after an attack onset are watched for a landing; an
        onset with no ``hit_landed`` in the window is a whiff. ``<= 0``
        disables whiff detection.
    """
    events: list[Event] = []
    hits_landed: list[Event] = []
    hits_taken: list[Event] = []
    attack_onsets: list[Event] = []

    prev: dict | None = None
    prev_player_dead = False
    prev_enemy_dead = False

    for i, state in enumerate(states):
        frame_index = int(_state_field(state, "frame_index", i))
        timestamp_sec = float(_state_field(state, "timestamp_sec", 0.0))
        round_index = int(_state_field(state, "round_index", 0))
        player_health = float(_state_field(state, "player_health", 0.0))
        enemy_mean = _mean_enemy_health(state)
        attacking = bool(_state_field(state, "attacking", False))
        domain_ready = bool(_state_field(state, "domain_ready", False))

        if prev is None:
            prev = {
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "round_index": round_index,
                "player_health": player_health,
                "enemy_mean": enemy_mean,
                "attacking": attacking,
                "domain_ready": domain_ready,
            }
            # A death already in progress on the first frame is not an event
            # we can attribute to a delta; leave the latches unset.
            continue

        # A round restarts the comparison baseline: the health reset back to
        # full must not look like an event.
        boundary = (
            round_index != prev["round_index"]
            or player_health > prev["player_health"] + 0.5
        )
        if not boundary:
            player_drop = prev["player_health"] - player_health
            enemy_drop = prev["enemy_mean"] - enemy_mean

            if enemy_drop >= health_drop_threshold:
                hit = Event(
                    type="hit_landed",
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    round_index=round_index,
                    detail={
                        "target": "enemy",
                        "health_before": round(prev["enemy_mean"], 4),
                        "health_after": round(enemy_mean, 4),
                        "damage": round(enemy_drop, 4),
                    },
                )
                events.append(hit)
                hits_landed.append(hit)

            if player_drop >= health_drop_threshold:
                hit = Event(
                    type="hit_taken",
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    round_index=round_index,
                    detail={
                        "target": "player",
                        "health_before": round(prev["player_health"], 4),
                        "health_after": round(player_health, 4),
                        "damage": round(player_drop, 4),
                    },
                )
                events.append(hit)
                hits_taken.append(hit)

            # Death episodes: capped to one event per death (fire on the
            # frame the health crosses the threshold, re-arm on recovery).
            if (
                not prev_player_dead
                and player_health <= death_threshold
                and prev["player_health"] > death_threshold
            ):
                events.append(
                    Event(
                        type="round_loss",
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        round_index=round_index,
                        detail={
                            "actor": "player",
                            "health_before": round(prev["player_health"], 4),
                            "health_after": round(player_health, 4),
                        },
                    )
                )
            if (
                not prev_enemy_dead
                and enemy_mean <= death_threshold
                and prev["enemy_mean"] > death_threshold
            ):
                events.append(
                    Event(
                        type="round_win",
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        round_index=round_index,
                        detail={
                            "actor": "enemy",
                            "health_before": round(prev["enemy_mean"], 4),
                            "health_after": round(enemy_mean, 4),
                        },
                    )
                )

        # Recovery clears the "dead" latch so a later death is a new event.
        if player_health > death_threshold:
            prev_player_dead = False
        if enemy_mean > death_threshold:
            prev_enemy_dead = False

        # HUD flash: domain deployment (domain_ready rising edge).
        if domain_ready and not prev["domain_ready"]:
            events.append(
                Event(
                    type="domain_alert",
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    round_index=round_index,
                    detail={"domain": "ready"},
                )
            )

        # Record attack onsets for the whiff pass below.
        if attacking and not prev["attacking"]:
            attack_onsets.append(
                Event(
                    type="whiff",
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    round_index=round_index,
                    detail={"attack_onset": True},
                )
            )

        prev = {
            "frame_index": frame_index,
            "timestamp_sec": timestamp_sec,
            "round_index": round_index,
            "player_health": player_health,
            "enemy_mean": enemy_mean,
            "attacking": attacking,
            "domain_ready": domain_ready,
        }
        if player_health <= death_threshold:
            prev_player_dead = True

    # ---- Post pass: punish (land without a return) & whiff (no landing) ----
    if punish_window_frames > 0:
        for hit in hits_landed:
            # A return on the same frame pair (>=) is a trade, not a punish --
            # the enemy banked damage too. A return strictly after the window
            # means the landing went unanswered => free damage for the player.
            returned = any(
                t.frame_index >= hit.frame_index
                and t.frame_index <= hit.frame_index + punish_window_frames
                for t in hits_taken
            )
            if not returned:
                events.append(
                    Event(
                        type="punish",
                        frame_index=hit.frame_index,
                        timestamp_sec=hit.timestamp_sec,
                        round_index=hit.round_index,
                        detail={**hit.detail, "hit_frame_index": hit.frame_index},
                    )
                )

    # A whiff is an attack onset that never lands. The landing on the onset
    # frame itself counts (the drop is read one frame after the attack start).
    if whiff_window_frames > 0:
        for onset in attack_onsets:
            landed = any(
                h.frame_index >= onset.frame_index
                and h.frame_index <= onset.frame_index + whiff_window_frames
                for h in hits_landed
            )
            if not landed:
                events.append(
                    Event(
                        type="whiff",
                        frame_index=onset.frame_index,
                        timestamp_sec=onset.timestamp_sec,
                        round_index=onset.round_index,
                        detail={"attack_onset": True},
                    )
                )

    # Deterministic order for the timeline and the report layer.
    events.sort(key=lambda e: (e.frame_index, EVENT_TYPES.index(e.type)))
    return events


__all__ = ["EVENT_TYPES", "Event", "detect_events"]