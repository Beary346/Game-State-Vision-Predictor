"""Tests for the Gold layer: state classification (winning/losing/stalemate).

The Gold layer consumes Silver feature JSONs plus human labels, trains a
variety of classifiers under MLflow, and picks the best one.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.pipeline.gold import (
    FEATURE_NAMES,
    MODEL_ZOO,
    STATE_LABELS,
    apply_corrections,
    classify_states,
    features_to_vector,
    generate_synthetic_dataset,
    load_labeled_dataset,
    predict,
    process_vod_report,
    rule_based_label,
    train_and_compare,
    train_gold_model,
)
from src.pipeline.silver import SilverFeatures

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def winning_features() -> dict:
    """A clearly winning state: the player is striking (attacking)."""
    return {
        "image_path": "/fake/screenshot.png",
        "player_health": 0.8,
        "player_position": [0.5, 0.5],
        "enemies": [
            {
                "bbox": [100, 200, 60, 60],
                "health": 0.3,
                "health_bar_bbox": [100, 170, 50, 8],
                "confidence": 0.9,
            }
        ],
        "attacking": True,
        "defending": False,
        "damage_indicator": False,
        "num_enemies": 1,
    }


@pytest.fixture
def losing_features() -> dict:
    """A clearly losing state: the player is being hit (damage flash)."""
    return {
        "image_path": "/fake/screenshot.png",
        "player_health": 0.2,
        "player_position": [0.5, 0.5],
        "enemies": [
            {
                "bbox": [100, 200, 60, 60],
                "health": 0.9,
                "health_bar_bbox": [100, 170, 50, 8],
                "confidence": 0.9,
            },
            {
                "bbox": [300, 200, 60, 60],
                "health": 0.8,
                "health_bar_bbox": [300, 170, 50, 8],
                "confidence": 0.9,
            },
        ],
        "attacking": False,
        "defending": True,
        "damage_indicator": True,
        "num_enemies": 2,
    }


@pytest.fixture
def stalemate_features() -> dict:
    """A neutral state: no attack, no damage flash — nothing is happening."""
    return {
        "image_path": "/fake/screenshot.png",
        "player_health": 0.5,
        "player_position": [0.5, 0.5],
        "enemies": [
            {
                "bbox": [100, 200, 60, 60],
                "health": 0.5,
                "health_bar_bbox": [100, 170, 50, 8],
                "confidence": 0.9,
            }
        ],
        "attacking": False,
        "defending": False,
        "damage_indicator": False,
        "num_enemies": 1,
    }


@pytest.fixture(scope="session")
def real_silver_json() -> dict | None:
    """Load a real Silver features JSON from data/silver/, if one exists."""
    silver_dir = Path(__file__).resolve().parents[1] / "data" / "silver"
    candidates = sorted(silver_dir.glob("*_silver.json"))
    if not candidates:
        return None
    with open(candidates[0]) as f:
        return json.load(f)


@pytest.fixture
def labeled_dataset(tmp_path) -> str:
    """A small synthetic labeled dataset (silver/ + labels.csv)."""
    return generate_synthetic_dataset(str(tmp_path / "ds"), num_samples=30, seed=7)


# ── Feature vectorization ────────────────────────────────────────────────────


class TestFeaturesToVector:
    def test_vector_shape_and_dtype(self, winning_features):
        vec = features_to_vector(winning_features)
        assert vec.shape == (len(FEATURE_NAMES),)
        assert vec.dtype == np.float32

    def test_feature_names_constant(self):
        assert len(FEATURE_NAMES) == 29
        assert FEATURE_NAMES[0] == "player_health"
        assert "health_ratio" in FEATURE_NAMES
        # NVIDIA-style crosses are present and after the base features.
        assert "player_health_x_health_ratio" in FEATURE_NAMES
        assert FEATURE_NAMES.index("player_health") < FEATURE_NAMES.index("player_health_x_health_ratio")

    def test_health_ratio_computation(self, winning_features):
        vec = features_to_vector(winning_features)
        expected_ratio = 0.8 / 0.3
        assert vec[FEATURE_NAMES.index("health_ratio")] == pytest.approx(expected_ratio)

    def test_zero_enemies(self):
        sf = {
            "image_path": "/p.png",
            "player_health": 0.9,
            "player_position": [0.5, 0.5],
            "enemies": [],
            "attacking": False,
            "defending": False,
            "damage_indicator": False,
            "num_enemies": 0,
        }
        vec = features_to_vector(sf)
        assert vec[FEATURE_NAMES.index("num_enemies")] == 0
        assert vec[FEATURE_NAMES.index("mean_enemy_health")] == 0.0
        assert vec[FEATURE_NAMES.index("min_enemy_health")] == 0.0

    def test_matches_silverfeatures_schema(self, winning_features):
        """features_to_vector must accept exactly what SilverFeatures serializes to."""
        sf = SilverFeatures(
            image_path=winning_features["image_path"],
            player_health=winning_features["player_health"],
            player_position=(0.5, 0.5),
            enemies=winning_features["enemies"],
            attacking=winning_features["attacking"],
            defending=winning_features["defending"],
            damage_indicator=winning_features["damage_indicator"],
            num_enemies=winning_features["num_enemies"],
        )
        vec = features_to_vector(sf.__dict__)
        assert vec.shape == (len(FEATURE_NAMES),)

    def test_real_silver_json_works(self, real_silver_json):
        if real_silver_json is None:
            pytest.skip("No real silver JSON found in data/silver/")
        vec = features_to_vector(real_silver_json)
        assert vec.shape == (len(FEATURE_NAMES),)
        assert np.isfinite(vec).all()

    def test_cross_features_are_multiplicative(self, winning_features):
        """NVIDIA-style crosses multiply their base parts (e.g. joy:
        player_health * health_ratio), so the model sees combinations."""
        vec = features_to_vector(winning_features)
        health = vec[FEATURE_NAMES.index("player_health")]
        ratio = vec[FEATURE_NAMES.index("health_ratio")]
        pos_x = vec[FEATURE_NAMES.index("player_position_x")]
        attacking = vec[FEATURE_NAMES.index("attacking")]
        n_enemies = vec[FEATURE_NAMES.index("num_enemies")]
        assert vec[FEATURE_NAMES.index("player_health_x_health_ratio")] == pytest.approx(health * ratio)
        assert vec[FEATURE_NAMES.index("player_health_x_player_position_x")] == pytest.approx(health * pos_x)
        assert vec[FEATURE_NAMES.index("attacking_x_health_ratio")] == pytest.approx(attacking * ratio)
        assert vec[FEATURE_NAMES.index("attacking_x_num_enemies")] == pytest.approx(attacking * n_enemies)

    def test_cooldowns_default_from_domain_ready(self):
        """Old 8-field dicts without per-slot cooldowns fall back to the
        domain_ready boolean they carried."""
        sf = {
            "player_health": 0.5,
            "player_position": [0.5, 0.5],
            "enemies": [{"health": 0.5}],
            "attacking": False,
            "defending": False,
            "damage_indicator": False,
            "num_enemies": 1,
            "domain_ready": True,
        }
        vec = features_to_vector(sf)
        assert vec[FEATURE_NAMES.index("cooldown_0")] == 1.0
        assert vec[FEATURE_NAMES.index("cooldown_3")] == 1.0
        sf["domain_ready"] = False
        vec = features_to_vector(sf)
        assert vec[FEATURE_NAMES.index("cooldown_0")] == 0.0

    def test_cooldowns_default_false_without_domain_ready(self):
        """Minimal hand-built dicts still produce a finite vector."""
        sf = {"player_health": 0.5, "player_position": [0.5, 0.5], "enemies": []}
        vec = features_to_vector(sf)
        assert vec.shape == (len(FEATURE_NAMES),)
        assert np.isfinite(vec).all()
        assert vec[FEATURE_NAMES.index("cooldown_0")] == 0.0


# ── Rule-based labeling ──────────────────────────────────────────────────────


class TestRuleBasedLabel:
    def test_winning_case(self, winning_features):
        assert rule_based_label(winning_features) == "winning"

    def test_losing_case(self, losing_features):
        assert rule_based_label(losing_features) == "losing"

    def test_stalemate_case(self, stalemate_features):
        assert rule_based_label(stalemate_features) == "stalemate"

    def test_damage_taken_shifts_toward_losing(self):
        sf = {
            "image_path": "/p.png",
            "player_health": 0.4,
            "player_position": [0.5, 0.5],
            "enemies": [{"health": 0.8}],
            "attacking": True,
            "defending": False,
            "damage_indicator": True,
            "num_enemies": 1,
        }
        # Player is attacking AND being hit -> being hit wins the overlap.
        assert rule_based_label(sf) == "losing"

    def test_defending_alone_is_stalemate(self):
        sf = {
            "player_health": 0.5,
            "player_position": [0.5, 0.5],
            "enemies": [{"health": 0.5}],
            "attacking": False,
            "defending": True,
            "damage_indicator": False,
            "num_enemies": 1,
        }
        # Guards don't decide the label: defensive posture is neutral.
        assert rule_based_label(sf) == "stalemate"

    def test_health_alone_does_not_decide(self):
        # Full health vs slivered enemy with no exchange at all is still
        # stalemate -- the label is initiative-only, health drives nothing.
        sf = {
            "player_health": 0.99,
            "player_position": [0.5, 0.5],
            "enemies": [{"health": 0.05}],
            "attacking": False,
            "defending": False,
            "damage_indicator": False,
            "num_enemies": 1,
        }
        assert rule_based_label(sf) == "stalemate"
        sf["player_health"] = 0.01
        assert rule_based_label(sf) == "stalemate"

    def test_deterministic(self, winning_features):
        assert rule_based_label(winning_features) == rule_based_label(winning_features)

    def test_label_is_always_valid(self):
        rng = np.random.default_rng(11)
        for _ in range(50):
            sf = {
                "image_path": "/p.png",
                "player_health": float(rng.uniform(0.05, 1.0)),
                "player_position": [0.5, 0.5],
                "enemies": [
                    {
                        "bbox": [100, 200, 60, 60],
                        "health": float(rng.uniform(0.05, 1.0)),
                        "health_bar_bbox": [100, 170, 50, 8],
                        "confidence": 0.9,
                    }
                ],
                "attacking": bool(rng.integers(0, 2)),
                "defending": bool(rng.integers(0, 2)),
                "damage_indicator": bool(rng.integers(0, 2)),
                "num_enemies": 1,
            }
            assert rule_based_label(sf) in STATE_LABELS

    def test_searching_when_no_enemies(self):
        sf = {
            "player_health": 0.7,
            "player_position": [0.5, 0.5],
            "enemies": [],
            "attacking": False,
            "defending": False,
            "damage_indicator": False,
            "num_enemies": 0,
        }
        assert rule_based_label(sf) == "searching"

    def test_won_on_enemy_death(self):
        sf = {
            "player_health": 0.8,
            "player_position": [0.5, 0.5],
            "enemies": [{"health": 0.0}, {"health": 0.4}],
            "attacking": False,
            "defending": False,
            "damage_indicator": False,
            "num_enemies": 1,
        }
        assert rule_based_label(sf) == "won"

    def test_lost_on_player_death(self):
        sf = {
            "player_health": 0.0,
            "player_position": [0.5, 0.5],
            "enemies": [{"health": 0.9}],
            "attacking": False,
            "defending": False,
            "damage_indicator": False,
            "num_enemies": 1,
        }
        assert rule_based_label(sf) == "lost"

    def test_player_death_beats_searching_and_won(self):
        """A dead player is 'lost' even if no enemy is on screen or the rule
        would call the trade a 'won' (both zeroed, player frames first)."""
        sf = {
            "player_health": 0.0,
            "player_position": [0.5, 0.5],
            "enemies": [],
            "attacking": False,
            "defending": False,
            "damage_indicator": False,
            "num_enemies": 0,
        }
        assert rule_based_label(sf) == "lost"
        sf["enemies"] = [{"health": 0.0}]
        assert rule_based_label(sf) == "lost"

    def test_death_moment_frames_only(self):
        """Health exactly zero is the death moment (won/lost); a sliver of HP
        on either side stays a normal fight state."""
        sf = {
            "player_health": 0.0,
            "player_position": [0.5, 0.5],
            "enemies": [{"health": 0.1}],
            "attacking": False,
            "defending": False,
            "damage_indicator": True,
            "num_enemies": 1,
        }
        assert rule_based_label(sf) == "lost"
        sf["player_health"] = 0.05  # alive again -> ordinary 'losing' frame
        assert rule_based_label(sf) == "losing"

    def test_enemy0_hp_override_corrects_rule_and_vector(self):
        """The labeler's ``enemy0_hp`` silver_override rewrites the first
        enemy's health, so a fixed misread moves the rule label (e.g. the
        visible enemy was actually dead) and the mined feature vector."""
        sf = {
            "player_health": 0.8,
            "player_position": [0.5, 0.5],
            "enemies": [{"health": 0.9}],
            "attacking": False,
            "defending": False,
            "damage_indicator": False,
            "num_enemies": 1,
        }
        scaffold = {"silver_override": {"enemy0_hp": 0.0}}
        corrected = apply_corrections(sf, scaffold)
        assert corrected["enemies"][0]["health"] == 0.0
        assert rule_based_label(corrected) == "won"
        vec = features_to_vector(corrected)
        assert vec[FEATURE_NAMES.index("min_enemy_health")] == 0.0
        assert vec[FEATURE_NAMES.index("mean_enemy_health")] == 0.0
        assert not np.isnan(vec).any()


# ── Synthetic labeled dataset ────────────────────────────────────────────────


class TestGenerateSyntheticDataset:
    def test_creates_silver_and_labels(self, labeled_dataset):
        ds = Path(labeled_dataset)
        silver_jsons = sorted((ds / "silver").glob("*.json"))
        assert len(silver_jsons) == 30
        assert (ds / "labels.csv").exists()

    def test_labels_are_valid(self, labeled_dataset):
        with open(Path(labeled_dataset) / "labels.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 30
        for row in rows:
            assert row["label"] in STATE_LABELS
            assert row["stem"]

    def test_csv_matches_embedded_labels(self, labeled_dataset):
        ds = Path(labeled_dataset)
        with open(ds / "labels.csv") as f:
            csv_labels = {row["stem"]: row["label"] for row in csv.DictReader(f)}
        for json_path in (ds / "silver").glob("*.json"):
            with open(json_path) as f:
                data = json.load(f)
            assert data["label"] == csv_labels[json_path.stem]

    def test_schema_matches_silverfeatures(self, labeled_dataset):
        """Generated JSONs must deserialize into SilverFeatures cleanly."""
        sample = next((Path(labeled_dataset) / "silver").glob("*.json"))
        with open(sample) as f:
            data = json.load(f)
        sf = SilverFeatures(**{k: v for k, v in data.items() if k != "label"})
        assert 0.0 <= sf.player_health <= 1.0
        assert sf.num_enemies >= 0

    def test_reproducible_with_seed(self, tmp_path):
        a = generate_synthetic_dataset(str(tmp_path / "a"), num_samples=5, seed=3)
        b = generate_synthetic_dataset(str(tmp_path / "b"), num_samples=5, seed=3)
        for json_a in sorted((Path(a) / "silver").glob("*.json")):
            json_b = Path(b) / "silver" / json_a.name
            with open(json_a) as fa, open(json_b) as fb:
                assert json.load(fa) == json.load(fb)
        assert (Path(a) / "labels.csv").read_text() == (Path(b) / "labels.csv").read_text()


# ── Dataset loading ──────────────────────────────────────────────────────────


class TestLoadLabeledDataset:
    def test_returns_expected_shapes(self, labeled_dataset):
        X, y, stems = load_labeled_dataset(labeled_dataset)
        assert X.shape == (30, len(FEATURE_NAMES))
        assert y.shape == (30,)
        assert len(stems) == 30
        assert set(y.tolist()) <= set(range(len(STATE_LABELS)))

    def test_csv_overrides_embedded_label(self, labeled_dataset):
        ds = Path(labeled_dataset)
        csv_path = ds / "labels.csv"
        rows = list(csv.DictReader(csv_path.open()))
        first_stem = rows[0]["stem"]
        new_label = "stalemate" if rows[0]["label"] != "stalemate" else "winning"
        rows[0]["label"] = new_label
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["stem", "label"])
            writer.writeheader()
            writer.writerows(rows)
        _, y, stems = load_labeled_dataset(labeled_dataset)
        idx = stems.index(first_stem)
        assert y[idx] == STATE_LABELS.index(new_label)

    def test_embedded_label_fallback(self, tmp_path, labeled_dataset):
        """Without labels.csv, embedded 'label' keys still load."""
        ds = Path(labeled_dataset)
        (ds / "labels.csv").unlink()
        X, y, _ = load_labeled_dataset(labeled_dataset)
        assert X.shape[0] == 30
        assert set(y.tolist()) <= set(range(len(STATE_LABELS)))

    def test_raises_on_unlabeled_sample(self, tmp_path):
        ds = tmp_path / "unlabeled"
        silver_dir = ds / "silver"
        silver_dir.mkdir(parents=True)
        with open(silver_dir / "frame_000.json", "w") as f:
            json.dump(
                {
                    "image_path": "/p.png",
                    "player_health": 0.5,
                    "player_position": [0.5, 0.5],
                    "enemies": [],
                    "attacking": False,
                    "defending": False,
                    "damage_indicator": False,
                    "num_enemies": 0,
                },
                f,
            )
        with pytest.raises(ValueError, match="frame_000"):
            load_labeled_dataset(str(ds))

    def test_raises_on_missing_root(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_labeled_dataset(str(tmp_path / "nope"))


# ── Training ─────────────────────────────────────────────────────────────────


class TestTrainGoldModel:
    def test_default_model_trains(self, labeled_dataset):
        result = train_gold_model(
            labeled_dataset,
            model_name="random_forest",
            model_params={"n_estimators": 20},
            experiment_name="test_gold_single",
            run_name="pytest_rf",
        )
        assert result["model_name"] == "random_forest"
        assert "run_id" in result
        assert 0.0 <= result["accuracy"] <= 1.0
        assert 0.0 <= result["macro_f1"] <= 1.0

    def test_confusion_matrix_artifact_logged(self, labeled_dataset):
        import mlflow

        result = train_gold_model(
            labeled_dataset,
            model_name="logistic_regression",
            experiment_name="test_gold_artifact",
            run_name="pytest_lr",
        )
        client = mlflow.tracking.MlflowClient()
        artifact_names = [a.path for a in client.list_artifacts(result["run_id"])]
        assert any("confusion_matrix" in name for name in artifact_names)
        assert any("classification_report" in name for name in artifact_names)

    def test_every_model_in_zoo_trains(self, labeled_dataset):
        """All six models must train and return sane metrics."""
        for name in MODEL_ZOO:
            params = {"n_estimators": 10, "max_depth": 2, "epochs": 5, "hidden_dim": 8}
            result = train_gold_model(
                labeled_dataset,
                model_name=name,
                model_params=params,
                experiment_name="test_gold_zoo",
                run_name=f"pytest_{name}",
            )
            assert result["model_name"] == name
            assert 0.0 <= result["macro_f1"] <= 1.0

    def test_raises_on_unlabeled_data(self, tmp_path):
        ds = tmp_path / "ds"
        silver_dir = ds / "silver"
        silver_dir.mkdir(parents=True)
        with open(silver_dir / "frame_000.json", "w") as f:
            json.dump(
                {
                    "image_path": "/p.png",
                    "player_health": 0.5,
                    "player_position": [0.5, 0.5],
                    "enemies": [],
                    "attacking": False,
                    "defending": False,
                    "damage_indicator": False,
                    "num_enemies": 0,
                },
                f,
            )
        with pytest.raises(ValueError):
            train_gold_model(str(ds), experiment_name="test_gold_error")


# ── Prediction ───────────────────────────────────────────────────────────────


class TestPredict:
    def _trained_model(self, labeled_dataset, model_name="random_forest"):
        import mlflow

        result = train_gold_model(
            labeled_dataset,
            model_name=model_name,
            model_params={"n_estimators": 20, "epochs": 10},
            experiment_name="test_gold_predict",
            run_name=f"predict_{model_name}",
        )
        model = mlflow.sklearn.load_model(result["model_path"])
        return model

    def test_predict_winning(self, labeled_dataset, winning_features):
        model = self._trained_model(labeled_dataset)
        label, probs = predict(winning_features, model)
        assert label == "winning"
        assert set(probs.keys()) == set(STATE_LABELS)
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)

    def test_predict_accepts_vector(self, labeled_dataset, winning_features):
        model = self._trained_model(labeled_dataset)
        vec = features_to_vector(winning_features)
        label, _ = predict(vec, model)
        assert label == "winning"

    def test_predict_with_mlp(self, labeled_dataset, winning_features):
        model = self._trained_model(labeled_dataset, model_name="pytorch_mlp")
        label, probs = predict(winning_features, model)
        assert label in STATE_LABELS
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)


# ── Per-frame state classification over a stream ─────────────────────────────


class TestClassifyStates:
    def _trained_model(self, labeled_dataset, model_name="random_forest"):
        import mlflow

        result = train_gold_model(
            labeled_dataset,
            model_name=model_name,
            model_params={"n_estimators": 20, "epochs": 10},
            experiment_name="test_gold_stream",
            run_name=f"stream_{model_name}",
        )
        return mlflow.sklearn.load_model(result["model_path"])

    def test_classify_states_adds_labels(
        self, labeled_dataset, winning_features, losing_features, stalemate_features
    ):
        model = self._trained_model(labeled_dataset)
        stream = [winning_features, stalemate_features, losing_features]
        out = classify_states(stream, model)
        assert len(out) == 3
        for row in out:
            state = row["state"]
            assert state["state_label"] in STATE_LABELS
            assert set(row["state_probs"].keys()) == set(STATE_LABELS)
            assert sum(row["state_probs"].values()) == pytest.approx(1.0, abs=1e-5)

    def test_classify_states_preserves_original_keys(self, labeled_dataset, winning_features):
        model = self._trained_model(labeled_dataset)
        out = classify_states([winning_features], model)
        state = out[0]["state"]
        for key in ("player_health", "enemies", "attacking", "num_enemies"):
            assert key in state
        assert out[0]["frame_index"] == winning_features.get("frame_index", 0)

    def test_rule_based_fallback_without_model(self, winning_features):
        out = classify_states([winning_features], model=None)
        assert out[0]["state"]["state_label"] == rule_based_label(winning_features)

    def test_empty_stream(self):
        assert classify_states([], model=None) == []


# ── Per-VOD Gold report (end-to-end) ─────────────────────────────────────────


class TestProcessVodReport:
    def _trained_model(self, labeled_dataset):
        import mlflow

        result = train_gold_model(
            labeled_dataset,
            model_name="random_forest",
            model_params={"n_estimators": 20},
            experiment_name="test_gold_vod_pipeline",
            run_name="pytest_vod_model",
        )
        return mlflow.sklearn.load_model(result["model_path"])

    def _stream(self, base_features, n=3):
        """Repeat one feature dict across a few frames with timestamps."""
        out = []
        for i in range(n):
            row = dict(base_features)
            row["frame_index"] = i
            row["timestamp_sec"] = i / 30.0
            out.append(row)
        return out

    def test_full_pipeline_runs(self, tmp_path, labeled_dataset, winning_features):
        model = self._trained_model(labeled_dataset)
        result = process_vod_report(
            states=self._stream(winning_features),
            model=model,
            vod_file="match.mp4",
            output_dir=str(tmp_path / "out"),
            experiment_name="test_gold_vod_pipeline",
            run_name="pytest_vod",
        )
        assert "run_id" in result
        assert result["vod_file"] == "match.mp4"
        assert 0.0 <= result["score"] <= 100.0
        assert result["n_frames"] == 3
        assert Path(result["timeline_json"]).exists()
        assert Path(result["stat_card"]).exists()

    def test_mlflow_metrics_logged(self, tmp_path, labeled_dataset, winning_features):
        model = self._trained_model(labeled_dataset)
        result = process_vod_report(
            states=self._stream(winning_features),
            model=model,
            vod_file="match.mp4",
            output_dir=str(tmp_path / "out"),
            experiment_name="test_gold_vod_metrics",
        )
        import mlflow

        run = mlflow.get_run(result["run_id"])
        assert float(run.data.metrics["n_frames"]) == 3
        assert "score" in run.data.metrics
        assert "mean_ocr_confidence" in run.data.metrics
        assert "total_events" in run.data.metrics

    def test_default_output_dir(self, tmp_path, labeled_dataset, winning_features):
        model = self._trained_model(labeled_dataset)
        result = process_vod_report(
            states=self._stream(winning_features),
            model=model,
            vod_file="match.mp4",
            experiment_name="test_gold_vod_default_dir",
        )
        assert Path(result["timeline_json"]).exists()
        assert "match" in result["timeline_json"]


# ── Model comparison ─────────────────────────────────────────────────────────


class TestTrainAndCompare:
    def test_leaderboard_ranks_all_models(self, labeled_dataset):
        df = train_and_compare(
            labeled_dataset,
            model_params={"n_estimators": 10, "max_depth": 2, "epochs": 5, "hidden_dim": 8},
            experiment_name="test_gold_compare",
        )
        assert len(df) == len(MODEL_ZOO)
        assert set(df["model_name"]) == set(MODEL_ZOO.keys())
        assert list(df["macro_f1"]) == sorted(df["macro_f1"], reverse=True)
        assert {"model_name", "accuracy", "macro_f1"}.issubset(df.columns)

    def test_leaderboard_run_logs_artifacts(self, labeled_dataset):
        import mlflow

        train_and_compare(
            labeled_dataset,
            model_params={"n_estimators": 10, "max_depth": 2, "epochs": 5, "hidden_dim": 8},
            experiment_name="test_gold_compare_artifacts",
        )
        runs = mlflow.search_runs(experiment_names=["test_gold_compare_artifacts"])
        leaderboard_run = runs[runs["tags.mlflow.runName"] == "leaderboard"]
        assert len(leaderboard_run) == 1
        run_id = leaderboard_run.iloc[0]["run_id"]
        client = mlflow.tracking.MlflowClient()
        artifact_names = [a.path for a in client.list_artifacts(run_id)]
        assert "leaderboard.csv" in artifact_names
        assert any("correlations" in name for name in artifact_names)
        assert any("tsne" in name for name in artifact_names)
