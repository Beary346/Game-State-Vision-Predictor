"""Tests for Gold reporting: score, timeline JSON, PIL stat card, per-VOD MLflow run.

AGENTS.md contract for the reporting side of the Gold layer:

- A quantified match report: event timeline (hits, whiffs, punishes, round
  outcomes, domain deployments), headline stats, and a score.
- A stat card generated with PIL (readable at screen-share zoom).
- Every processed VOD is an MLflow run with params (VOD file, event
  thresholds), metrics (event counts, mean OCR confidence) and artifacts
  (timeline JSON + stat card).
"""

import json
import os
from pathlib import Path

import mlflow
import pytest
from PIL import Image

from src.pipeline.events_gold import detect_events
from src.pipeline.renderer_reportcard import render_stat_card
from src.pipeline.report import (
    build_headline,
    build_timeline,
    compute_score,
    generate_report,
    log_vod_report,
    mean_ocr_confidence,
)

# ── State helpers ────────────────────────────────────────────────────────────


def _state(
    frame_index: int = 0,
    player_health: float = 1.0,
    enemy_healths: list[float] | None = None,
    attacking: bool = False,
    domain_ready: bool = False,
    round_index: int = 0,
    ocr_confidence: float = 0.9,
) -> dict:
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
        "defending": False,
        "damage_indicator": False,
        "num_enemies": len(enemy_healths),
        "frame_index": frame_index,
        "timestamp_sec": float(frame_index) / 30.0,
        "clock_sec": 60.0 - frame_index,
        "domain_ready": domain_ready,
        "ocr_confidence": ocr_confidence,
        "round_index": round_index,
    }


@pytest.fixture
def match_states() -> list[dict]:
    """A short match: hits both ways, a punish, a whiff, a domain, two rounds."""
    states = [
        _state(0, player_health=1.0),                                      # round 0
        _state(1, player_health=1.0, enemy_healths=[0.8], attacking=True),
        _state(2, player_health=1.0, enemy_healths=[0.5]),                 # hit landed -> punish
        _state(3, player_health=0.75, enemy_healths=[0.5]),                # hit taken
        _state(4, player_health=0.75, enemy_healths=[0.2]),                # hit landed
        _state(5, player_health=0.30, enemy_healths=[0.2]),                # hit taken
        _state(6, player_health=0.02, enemy_healths=[0.2], round_index=0),  # round loss
        _state(7, player_health=1.0, enemy_healths=[0.8], round_index=1),  # round 1 starts
        _state(8, player_health=1.0, enemy_healths=[0.8], attacking=True),
        _state(9, player_health=1.0, enemy_healths=[0.8], attacking=True),  # whiff
        _state(10, player_health=1.0, enemy_healths=[0.0], domain_ready=True),  # punish + enemy death + domain
        _state(11, player_health=1.0, enemy_healths=[0.0]),
    ]
    return states


@pytest.fixture
def match_events(match_states: list[dict]) -> list:
    return detect_events(match_states)


# ── Score ────────────────────────────────────────────────────────────────────


class TestScore:
    def test_score_in_range(self, match_states, match_events):
        headline = build_headline(match_events)
        score = compute_score(headline)
        assert 0.0 <= score <= 100.0

    def test_more_hits_mean_higher_score(self):
        base = compute_score(build_headline([]))
        better = compute_score({"hit_landed": 5, "hit_taken": 0, "punish": 1, "whiff": 0})
        assert better > base

    def test_round_loss_weighs_heavily(self):
        losses = compute_score({"hit_landed": 4, "hit_taken": 1, "punish": 0, "whiff": 0, "round_loss": 2})
        wins = compute_score({"hit_landed": 4, "hit_taken": 1, "punish": 0, "whiff": 0, "round_win": 2})
        assert losses < wins

    def test_score_clamped_at_bounds(self):
        assert compute_score({"hit_landed": 0, "hit_taken": 100, "punish": 0, "whiff": 100, "round_loss": 8}) == 0.0
        assert compute_score({"hit_landed": 100, "hit_taken": 0, "punish": 100, "whiff": 0, "round_win": 5}) == 100.0

    def test_accepts_events_directly(self, match_events):
        score_events = compute_score(match_events)
        assert score_events == compute_score(build_headline(match_events))

    def test_headline_plural_names_are_scored(self):
        """Headline dicts use plural keys (hits_landed) and must be scored,
        not treated as zero. A single round loss outweighs any small score."""
        headline = {
            "hits_landed": 4,
            "hits_taken": 5,
            "punishes": 2,
            "whiffs": 2,
            "round_wins": 1,
            "round_losses": 2,
            "domain_alerts": 1,
            "total_events": 17,
            "damage_dealt": 1.35,
            "damage_taken": 1.9,
        }
        assert compute_score(headline) == pytest.approx(38.0)

    def test_round_outcomes_move_the_score(self):
        won = compute_score({"round_wins": 2, "round_losses": 0})
        lost = compute_score({"round_wins": 0, "round_losses": 2})
        assert won == pytest.approx(70.0)
        assert lost == pytest.approx(26.0)


# ── Headline stats ───────────────────────────────────────────────────────────


class TestHeadline:
    def test_counts_every_event_type(self, match_events):
        h = build_headline(match_events)
        for key in ("hits_landed", "hits_taken", "punishes", "whiffs", "round_wins", "round_losses", "domain_alerts"):
            assert key in h
        assert isinstance(h["hits_landed"], int)

    def test_zero_defaults(self):
        h = build_headline([])
        assert all(v == 0 for v in h.values())

    def test_damage_summed_from_details(self, match_events):
        h = build_headline(match_events)
        assert h["damage_dealt"] > 0.0
        assert h["damage_taken"] > 0.0


# ── Timeline JSON ────────────────────────────────────────────────────────────


class TestBuildTimeline:
    def test_core_keys(self, match_states, match_events):
        tl = build_timeline(states=match_states, events=match_events, vod_name="match.mp4")
        for key in ("vod_file", "score", "mean_ocr_confidence", "headline", "rounds", "events", "state_curve"):
            assert key in tl

    def test_state_curve_matches_frames(self, match_states, match_events):
        tl = build_timeline(states=match_states, events=match_events, vod_name="x.mp4")
        curve = tl["state_curve"]
        assert len(curve) == len(match_states)
        assert curve[0]["player_health"] == match_states[0]["player_health"]
        assert set(curve[0].keys()) >= {"timestamp_sec", "frame_index", "round_index", "player_health"}

    def test_events_serialized(self, match_states, match_events):
        tl = build_timeline(states=match_states, events=match_events, vod_name="x.mp4")
        assert tl["events"] == [
            {
                "type": e.type,
                "frame_index": e.frame_index,
                "timestamp_sec": e.timestamp_sec,
                "round_index": e.round_index,
                "detail": e.detail,
            }
            for e in match_events
        ]

    def test_rounds_aggregate(self, match_states, match_events):
        tl = build_timeline(states=match_states, events=match_events, vod_name="x.mp4")
        assert len(tl["rounds"]) == 2
        first = tl["rounds"][0]
        assert first["round_index"] == 0
        assert first["damage_taken"] > 0.0
        assert first["damage_dealt"] > 0.0

    def test_json_round_trip(self, match_states, match_events, tmp_path):
        tl = build_timeline(states=match_states, events=match_events, vod_name="x.mp4")
        p = tmp_path / "timeline.json"
        p.write_text(json.dumps(tl))
        assert json.loads(p.read_text()) == tl


# ── Mean OCR confidence ──────────────────────────────────────────────────────


class TestMeanOcr:
    def test_averages_over_states(self, match_states):
        assert mean_ocr_confidence(match_states) == pytest.approx(0.9)

    def test_empty_stream(self):
        assert mean_ocr_confidence([]) == 0.0


# ── Full report + stat card ──────────────────────────────────────────────────


class TestGenerateReport:
    def test_write_timeline_and_stat_card(self, match_states, match_events, tmp_path):
        result = generate_report(
            states=match_states,
            events=match_events,
            vod_name="match.mp4",
            output_dir=str(tmp_path),
        )
        assert Path(result["timeline_json"]).exists()
        assert Path(result["stat_card"]).exists()
        assert 0.0 <= result["score"] <= 100.0

    def test_timeline_json_valid(self, match_states, match_events, tmp_path):
        result = generate_report(
            states=match_states,
            events=match_events,
            vod_name="match.mp4",
            output_dir=str(tmp_path),
        )
        with open(result["timeline_json"]) as f:
            data = json.load(f)
        assert data["vod_file"] == "match.mp4"
        assert data["score"] == result["score"]

    def test_timeline_serializable_on_hello_world(self, match_states, match_events, tmp_path):
        result = generate_report(
            states=match_states,
            events=match_events,
            vod_name="match.mp4",
            output_dir=str(tmp_path),
        )
        # The timeline JSON must be re-importable by the notebook / web UI.
        with open(result["timeline_json"]) as f:
            json.load(f)


class TestStatCard:
    def test_pil_image_rendered(self, match_states, match_events, tmp_path):
        card_path = render_stat_card(
            states=match_states,
            events=match_events,
            vod_name="match.mp4",
            output_path=str(tmp_path / "card.png"),
        )
        img = Image.open(card_path)
        assert img.format == "PNG"
        assert img.width >= 1000
        assert img.height >= 600

    def test_stat_card_wide_for_screen_share(self, match_states, match_events, tmp_path):
        card_path = render_stat_card(
            states=match_states,
            events=match_events,
            output_path=str(tmp_path / "card.png"),
        )
        img = Image.open(card_path)
        assert img.width / img.height > 1.4

    def test_stat_card_written_by_report(self, match_states, match_events, tmp_path):
        result = generate_report(
            states=match_states,
            events=match_events,
            vod_name="match.mp4",
            output_dir=str(tmp_path),
        )
        img = Image.open(result["stat_card"])
        assert img.width > 0


# ── Per-VOD MLflow run ───────────────────────────────────────────────────────


@pytest.fixture
def tracking_uri(tmp_path) -> str:
    uri = f"sqlite:///{tmp_path / 'vod_mlflow.db'}"
    os.environ["MLFLOW_TRACKING_URI"] = uri
    yield uri
    os.environ.pop("MLFLOW_TRACKING_URI", None)


class TestLogVodReport:
    def test_run_logs_params_metrics_artifacts(self, match_states, match_events, tracking_uri, tmp_path):
        result = log_vod_report(
            states=match_states,
            events=match_events,
            vod_name="match.mp4",
            output_dir=str(tmp_path / "report"),
            experiment_name="test_gold_vod",
            event_kwargs={"health_drop_threshold": 0.06, "punish_window_frames": 5},
        )
        assert "run_id" in result
        run = mlflow.get_run(result["run_id"])
        # Params: VOD file + event thresholds.
        assert run.data.params["vod_file"] == "match.mp4"
        assert run.data.params["health_drop_threshold"] == "0.06"
        assert run.data.params["punish_window_frames"] == "5"
        # Metrics: event counts + mean OCR confidence + score.
        assert float(run.data.metrics["hits_landed"]) > 0
        assert float(run.data.metrics["punishes"]) > 0
        assert float(run.data.metrics["mean_ocr_confidence"]) == pytest.approx(0.9)
        assert "score" in run.data.metrics
        # Artifacts: timeline JSON + stat card.
        artifacts = [a.path for a in mlflow.tracking.MlflowClient().list_artifacts(result["run_id"])]
        assert any(name == "timeline.json" for name in artifacts)
        assert any("stat_card" in name for name in artifacts)

    def test_artifacts_exist_on_disk(self, match_states, match_events, tracking_uri, tmp_path):
        result = log_vod_report(
            states=match_states,
            events=match_events,
            vod_name="match.mp4",
            output_dir=str(tmp_path / "report"),
            experiment_name="test_gold_vod_artifacts",
        )
        assert Path(result["timeline_json"]).exists()
        assert Path(result["stat_card"]).exists()

    def test_empty_stream_logs_zero_metrics(self, tracking_uri, tmp_path):
        result = log_vod_report(
            states=[],
            events=[],
            vod_name="empty.mp4",
            output_dir=str(tmp_path / "report"),
            experiment_name="test_gold_vod_empty",
        )
        run = mlflow.get_run(result["run_id"])
        assert float(run.data.metrics["total_events"]) == 0