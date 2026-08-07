"""Tests for the labeling scaffold tooling: exclude/context/silver_override.

The scaffolds are the reviewer-facing copy of Silver features. These tests
cover the three new labeling powers (remove frames, structured context,
correct misread features) plus the existing flow (init -> label -> export ->
bootstrap rule refresh).
"""

import csv
import json
from pathlib import Path

import pytest

from src.pipeline import labeling
from src.pipeline.gold import (
    STATE_LABELS,
    apply_corrections,
    load_labeled_dataset,
    rule_based_label,
)

SILVER_SAMPLE = {
    "image_path": "/fake/screenshot.png",
    "player_health": 0.8,
    "player_position": [0.5, 0.5],
    "enemies": [{"bbox": [100, 200, 60, 60], "health": 0.3, "health_bar_bbox": [100, 170, 50, 8], "confidence": 0.9}],
    "attacking": True,
    "defending": False,
    "damage_indicator": False,
    "num_enemies": 1,
    "ability_cooldowns": [1.0, 0.5, 0.0, 1.0],
    "domain_ready": False,
    "player_ragdoll": False,
    "enemy_ragdoll": False,
    "player_ultimate": False,
    "enemy_ultimate": False,
    "domain_active": False,
    "domain_activating": False,
}


@pytest.fixture
def silver_dir(tmp_path) -> Path:
    d = tmp_path / "data" / "silver"
    d.mkdir(parents=True)
    for i in range(4):
        features = dict(SILVER_SAMPLE)
        features["player_health"] = 0.5 + 0.1 * i
        with open(d / f"vod_frame_{i:06d}_silver.json", "w") as f:
            json.dump(features, f)
    return d


@pytest.fixture
def labeling_dir(tmp_path) -> Path:
    return tmp_path / "data" / "labeling"


def test_init_creates_scaffolds_with_all_reviewer_fields(silver_dir, labeling_dir):
    res = labeling.init_labeling_files(silver_dir, labeling_dir)
    assert res == {"created": 4, "existing": 0, "total": 4}
    scaffold = labeling.load_scaffold(labeling_dir, "vod_frame_000000")
    assert scaffold["label"] is None
    assert scaffold["skip"] is False
    assert scaffold["exclude"] is False
    assert all(v is None for v in scaffold["context"].values())
    assert scaffold["silver_override"] == {}
    assert scaffold["rule_label"] == "winning"  # attacking, enemy present


def test_exclude_drops_from_export_and_training(silver_dir, labeling_dir):
    labeling.init_labeling_files(silver_dir, labeling_dir)
    for stem, payload in labeling.iter_scaffolds(labeling_dir, load=True):
        payload["label"] = "winning"
        labeling.save_scaffold(labeling_dir, stem, payload)

    labeling.load_scaffold(labeling_dir, "vod_frame_000001")["exclude"] = True
    payload = labeling.load_scaffold(labeling_dir, "vod_frame_000001")
    payload["exclude"] = True
    labeling.save_scaffold(labeling_dir, "vod_frame_000001", payload)

    out = labeling.export_labels_csv(labeling_dir, "labels.csv")
    assert out["rows"] == 3

    summ = labeling.summary(labeling_dir)
    assert summ["total"] == 4
    assert summ["excluded"] == 1
    assert summ["labeled"] == 3

    with open("labels.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert "vod_frame_000001" not in {r["stem"] for r in rows}


def test_context_keys_are_normalized_on_load(silver_dir, labeling_dir):
    labeling.init_labeling_files(silver_dir, labeling_dir)
    payload = labeling.load_scaffold(labeling_dir, "vod_frame_000000")
    assert set(payload["context"]) == set(labeling.CONTEXT_KEYS)
    assert all(v is None for v in payload["context"].values())


def test_valid_label_accepts_six_states():
    for label in STATE_LABELS:
        assert labeling.valid_label(label) == label
    assert labeling.valid_label("wining") is None
    assert labeling.valid_label(None) is None


def test_refresh_rule_uses_corrected_features(silver_dir, labeling_dir):
    labeling.init_labeling_files(silver_dir, labeling_dir)
    # Frame 000000 reads 'winning' (attacking). Correct the misread: force
    # attacking off + damage on -> losing.
    payload = labeling.load_scaffold(labeling_dir, "vod_frame_000000")
    payload["silver_override"] = {"attacking": False, "damage_indicator": True}
    labeling.save_scaffold(labeling_dir, "vod_frame_000000", payload)

    res = labeling.refresh_rule_labels(labeling_dir)
    assert res["refreshed"] == 1
    refreshed = labeling.load_scaffold(labeling_dir, "vod_frame_000000")
    assert rule_based_label(apply_corrections(refreshed, refreshed)) == "losing"


def test_context_overrides_features_at_training(silver_dir, labeling_dir, tmp_path):
    """A labeler-confirmed context value replaces a misread Silver feature
    inside the training loop (load_labeled_dataset applies corrections)."""
    from src.pipeline.gold import FEATURE_NAMES

    labeling.init_labeling_files(silver_dir, labeling_dir)
    payload = labeling.load_scaffold(labeling_dir, "vod_frame_000000")
    payload["context"] = {"player_ragdolled": True, "enemy_ragdolled": True}
    payload["label"] = "losing"
    labeling.save_scaffold(labeling_dir, "vod_frame_000000", payload)
    for stem, p in labeling.iter_scaffolds(labeling_dir, load=True):
        if stem != "vod_frame_000000":
            p["label"] = "stalemate"
            labeling.save_scaffold(labeling_dir, stem, p)
    labeling.export_labels_csv(labeling_dir, tmp_path / "data" / "labels.csv")

    X, _, stems = load_labeled_dataset(str(tmp_path / "data"))
    idx = stems.index("vod_frame_000000")
    assert X[idx, FEATURE_NAMES.index("player_ragdoll")] == 1.0
    assert X[idx, FEATURE_NAMES.index("enemy_ragdoll")] == 1.0
    # Other frames keep their Silver defaults (not ragdolled).
    other = next(i for i, s in enumerate(stems) if s != "vod_frame_000000")
    assert X[other, FEATURE_NAMES.index("player_ragdoll")] == 0.0


def test_apply_corrections_combines_override_and_context():
    features = dict(SILVER_SAMPLE)
    scaffold = {
        "silver_override": {"player_health": 0.42},
        "context": {"player_ragdolled": False, "enemy_ragdolled": True},
        "label": None,
    }
    corrected = apply_corrections(features, scaffold)
    assert corrected["player_health"] == 0.42
    assert corrected["player_ragdoll"] is False
    assert corrected["enemy_ragdoll"] is True
    # Unrelated features untouched.
    assert corrected["attacking"] is True


def test_apply_corrections_noop_without_scaffold():
    assert apply_corrections(dict(SILVER_SAMPLE), None) == SILVER_SAMPLE