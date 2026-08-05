"""Tests for the new HUD heuristic Silver layer (src/classifier_silver.py).

Every test is
deterministic: seeded synthetic frames are rendered with render_frame_and_state and
classified read back with classify_frame, so failures below double as bugs surface
instead of random noise.

The explicit "known read accuracy" contract lives in
TestHUDReadAccuracyOnSeededFrames: health fills must match within tolerance and
the clock OCR must read exactly on every seed.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.classifier_silver import (
    ABILITY_ROW_REGION,
    CLOCK_REGION,
    DAMAGE_FLASH_REGION,
    HEALTH_BAR_TOLERANCE,
    PLAYER_HEALTH_BAR,
    STATE_DOT_REGION,
    H,
    SilverFeatures,
    W,
    aggregate_rounds,
    assign_round_indices,
    classify_frame,
    process_frames,
    read_ability_indicators,
    read_clock_ocr,
    read_damage_flash,
    read_health_fill,
    read_state_dot,
    render_frame_and_state,
    simulate_match,
)


class TestHUDReadAccuracyOnSeededFrames:
    """Seeded, deterministic read-accuracy contract for HUD sub-readers."""

    def test_player_health_fill_accuracy(self):
        for health in (0.0, 0.15, 0.3, 0.8, 1.0):
            frame, _ = render_frame_and_state(player_health=health, enemy_healths=[0.5], seed=1)
            fill = read_health_fill(frame, PLAYER_HEALTH_BAR)[0]
            assert abs(fill - health) <= HEALTH_BAR_TOLERANCE, (fill, health)

    def test_clock_ocr_is_exact_across_digits(self):
        for seconds in (0, 9, 17, 42, 60, 99):
            frame, _ = render_frame_and_state(clock_sec=float(seconds), seed=1)
            read, conf = read_clock_ocr(frame, CLOCK_REGION)
            assert read == float(seconds), (read, seconds)
            assert conf >= 0.8

    def test_state_dot_attacking(self):
        frame, _ = render_frame_and_state(attacking=True, defending=False, seed=1)
        assert read_state_dot(frame, STATE_DOT_REGION) == (True, False)

    def test_state_dot_defending(self):
        frame, _ = render_frame_and_state(attacking=False, defending=True, seed=1)
        assert read_state_dot(frame, STATE_DOT_REGION) == (False, True)

    def test_state_dot_neutral(self):
        frame, _ = render_frame_and_state(attacking=False, defending=False, seed=1)
        assert read_state_dot(frame, STATE_DOT_REGION) == (False, False)

    def test_damage_flash_on(self):
        frame, _ = render_frame_and_state(damaged=True, seed=1)
        assert read_damage_flash(frame, DAMAGE_FLASH_REGION) is True

    def test_damage_flash_off(self):
        frame, _ = render_frame_and_state(damaged=False, seed=1)
        assert read_damage_flash(frame, DAMAGE_FLASH_REGION) is False

    def test_ability_row_fully_ready_is_domain(self):
        frame, _ = render_frame_and_state(
            domain_ready=True, ability_ready=[True, True, True, True], seed=1
        )
        ready, _ = read_ability_indicators(frame, ABILITY_ROW_REGION)
        assert ready == [True, True, True, True]

    def test_ability_row_partial_not_domain(self):
        frame, _ = render_frame_and_state(
            domain_ready=True, ability_ready=[True, True, False, True], seed=1
        )
        ready, _ = read_ability_indicators(frame, ABILITY_ROW_REGION)
        assert ready != [True] * 4


class TestClassifyRoundTrip:
    """Heuristic reader must recover the seeded ground truth exactly."""

    def test_frame_round_trip_single_enemy(self):
        for seed in range(12):
            frame, state = render_frame_and_state(enemy_healths=[0.5], domain_ready=None, seed=seed)
            out = classify_frame(frame)
            assert abs(out.player_health - state["player_health"]) <= HEALTH_BAR_TOLERANCE
            assert out.attacking == state["attacking"]
            assert out.defending == state["defending"]
            assert out.damage_indicator == state["damage_indicator"]
            assert out.domain_ready == state["domain_ready"]
            assert out.num_enemies == len(state["enemies"])
            assert out.player_position[0] == pytest.approx(0.5, abs=0.02)

    def test_clock_and_health_roundtrip_from_simulated_match(self):
        for entry in simulate_match(num_frames=40, seed=3):
            out = classify_frame(entry["frame"])
            assert out.clock_sec == pytest.approx(entry["clock_sec"], abs=1.0)
            assert 0.0 <= out.player_health <= 1.0

    def test_wrong_shape_rejected(self):
        with pytest.raises(AssertionError):
            classify_frame(np.zeros((W, H, 3), dtype=np.uint8))

    def test_low_resolution_rejected(self):
        with pytest.raises(AssertionError):
            classify_frame(np.zeros((360, 640, 3), dtype=np.uint8))


class TestAggregateRounds:
    def test_simulated_match_rounds_are_bundled(self):
        entries = simulate_match(num_frames=40, seed=3)
        states = [classify_frame(e["frame"]) for e in entries]
        tagged, summaries = aggregate_rounds(states)
        assert [s.round_index for s in tagged] == sorted(s.round_index for s in tagged)
        assert [r.num_frames for r in summaries] == [15, 15, 10]
        assert [r.round_index for r in summaries] == [0, 1, 2]

    def test_round_summary_health_stats(self):
        entries = simulate_match(num_frames=40, seed=3)
        states = [classify_frame(e["frame"]) for e in entries]
        tagged, summaries = aggregate_rounds(states)
        for summary in summaries:
            assert 0.0 <= summary.min_player_health <= 1.0
            assert summary.num_frames == len(
                [s for s in tagged if s.round_index == summary.round_index]
            )

    def test_assign_round_indices_catches_health_reset(self):
        states = [
            SilverFeatures(player_health=0.9, clock_sec=60.0, timestamp_sec=0.0),
            SilverFeatures(player_health=0.4, clock_sec=50.0, timestamp_sec=1.0),
            SilverFeatures(player_health=1.0, clock_sec=60.0, timestamp_sec=2.0),
        ]
        tagged = assign_round_indices(states)
        assert [s.round_index for s in tagged] == [0, 0, 1]


class TestProcessFrames:
    def test_processes_bronze_frames_and_writes_artifacts(self, tmp_path):
        bronze_dir = tmp_path / "bronze"
        silver_dir = tmp_path / "silver"
        bronze_dir.mkdir()
        for i, entry in enumerate(simulate_match(num_frames=5, seed=7)):
            png = bronze_dir / f"match_00000_bronze_frame_{i:06d}.png"
            cv2.imwrite(str(png), cv2.cvtColor(entry["frame"], cv2.COLOR_RGB2BGR))
            with open(png.with_suffix(".json"), "w", encoding="utf-8") as fh:
                json.dump({"frame_index": i, "timestamp_sec": i / 30.0}, fh)

        result = process_frames(bronze_dir, silver_dir)
        assert len(result["frames"]) == 5
        assert [r["num_frames"] for r in result["rounds"]] == [5]
        assert Path(result["rounds_json"]).exists()
        for artifact in result["frames"]:
            with open(artifact["features_json"], "r", encoding="utf-8") as fh:
                data = json.load(fh)
            sf = SilverFeatures(**data)
            assert isinstance(sf.player_health, float)
