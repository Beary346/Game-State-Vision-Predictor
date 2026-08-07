"""Tests for the train -> register -> use-on-unseen-VOD workflow.

AGENTS.md promises the state-reader becomes a versioned MLflow model
(Staging -> Production) and that processed VODs become MLflow runs. These tests
cover the two convenience bridges that make that workflow actually runnable:

1. ``register_best_model`` — after ``train_and_compare``, promote the
   leaderboard winner straight into the registry (Production).
2. ``infer_new_vod`` — run a *brand-new* VOD (never seen in training) through
   Bronze -> Silver -> Gold classification + report, using the registered
   production state-reader (or the rule-based fallback when none exists).
"""

import csv
import os
from pathlib import Path

import cv2
import mlflow
import numpy as np
import pandas as pd
import pytest

from src.pipeline.gold import (
    FEATURE_NAMES,
    H,
    W,
    generate_synthetic_dataset,
    infer_new_vod,
    register_best_model,
)

REGISTERED_NAME = "state_reader"


@pytest.fixture
def tracking_uri(tmp_path) -> str:
    uri = f"sqlite:///{tmp_path / 'usage.db'}"
    os.environ["MLFLOW_TRACKING_URI"] = uri
    os.environ["MLFLOW_REGISTRY_URI"] = uri
    # ensure_tracking_uri() only honors the env var and never calls
    # set_tracking_uri, and mlflow caches the store + active experiment id
    # across tests: pin both explicitly so a fresh DB is used every time.
    mlflow.set_tracking_uri(uri)
    yield uri
    # Reset so later test modules that resolve the URI from env vars (e.g.
    # test_model_registry.py) are not pinned to this test's sqlite DB. The
    # fluent module caches both the tracking URI and the active experiment id,
    # so also clear the experiment cache or mlflow keeps resolving the stale
    # experiment that only exists in this test's database.
    mlflow.set_tracking_uri(None)
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    os.environ.pop("MLFLOW_REGISTRY_URI", None)
    from mlflow.tracking import fluent

    fluent._active_experiment_id = None
    fluent.MLFLOW_EXPERIMENT_ID.unset()


@pytest.fixture
def cleanup_registry(tracking_uri):
    yield
    client = mlflow.tracking.MlflowClient()
    for mv in client.search_model_versions(f"name = '{REGISTERED_NAME}'"):
        client.delete_model_version(REGISTERED_NAME, mv.version)
    from mlflow.exceptions import MlflowException

    try:
        client.get_registered_model(REGISTERED_NAME)
    except MlflowException:
        pass
    else:
        client.delete_registered_model(REGISTERED_NAME)


@pytest.fixture
def logged_model_path(tracking_uri) -> str:
    """A tiny trained classifier logged to a fresh MLflow run (returns model URI)."""
    import mlflow.sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, (40, len(FEATURE_NAMES)))
    y = np.where(X[:, 5] > X[:, 4], 1, 0)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X, y)
    mlflow.set_experiment("test_model_usage_log")
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(model, "model")
    return info.model_uri


@pytest.fixture
def tiny_vod(tmp_path):
    """A short synthetic VOD, cheap to process end to end."""
    vod_path = tmp_path / "unseen_match.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    out = cv2.VideoWriter(str(vod_path), fourcc, fps, (W, H))
    for i in range(90):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            f"Frame {i}",
            (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            3,
        )
        out.write(frame)
    out.release()
    return str(vod_path)


class TestRegisterBestModel:
    def test_registers_winner_and_promotes(self, logged_model_path, cleanup_registry):
        leaderboard = pd.DataFrame(
            [
                {
                    "model_name": "xgboost",
                    "run_id": "r1",
                    "accuracy": 0.9,
                    "macro_f1": 0.87,
                    "weighted_f1": 0.89,
                    "model_path": logged_model_path,
                },
                {
                    "model_name": "random_forest",
                    "run_id": "r2",
                    "accuracy": 0.8,
                    "macro_f1": 0.76,
                    "weighted_f1": 0.78,
                    "model_path": logged_model_path,
                },
            ]
        )
        result = register_best_model(leaderboard, name=REGISTERED_NAME)
        assert result["name"] == REGISTERED_NAME
        assert result["stage"] == "Production"
        assert result["tags"]["model_name"] == "xgboost"
        assert result["tags"]["accuracy"] == "0.9"

    def test_raises_on_empty_leaderboard(self, tracking_uri):
        with pytest.raises(ValueError, match="leaderboard"):
            register_best_model(pd.DataFrame(), name=REGISTERED_NAME)


class TestInferNewVod:
    def test_runs_end_to_end_rule_based(self, tiny_vod, tracking_uri, tmp_path):
        """Without a registered model, inference falls back to the rules."""
        result = infer_new_vod(
            tiny_vod,
            output_dir=str(tmp_path / "report"),
            experiment_name="test_gold_usage",
            frame_interval=30,
        )
        assert "run_id" in result
        assert result["n_frames"] == 3
        assert Path(result["timeline_json"]).exists()
        assert Path(result["stat_card"]).exists()
        assert 0.0 <= result["score"] <= 100.0

    def test_uses_registered_production_model(
        self, tiny_vod, tracking_uri, tmp_path, logged_model_path, cleanup_registry
    ):
        """A registered state_reader is picked up automatically."""
        register_best_model(
            pd.DataFrame(
                [
                    {
                        "model_name": "logistic_regression",
                        "run_id": "r1",
                        "accuracy": 1.0,
                        "macro_f1": 1.0,
                        "weighted_f1": 1.0,
                        "model_path": logged_model_path,
                    }
                ]
            ),
            name=REGISTERED_NAME,
        )
        result = infer_new_vod(
            tiny_vod,
            output_dir=str(tmp_path / "report"),
            experiment_name="test_gold_usage",
            frame_interval=30,
        )
        assert result["n_frames"] == 3
        assert Path(result["timeline_json"]).exists()

    def test_explicit_model_path_wins(self, tiny_vod, tracking_uri, tmp_path, logged_model_path):
        import mlflow.sklearn

        model = mlflow.sklearn.load_model(logged_model_path)
        result = infer_new_vod(
            tiny_vod,
            model=model,
            output_dir=str(tmp_path / "report"),
            experiment_name="test_gold_usage",
            frame_interval=30,
        )
        assert result["n_frames"] == 3
        assert Path(result["stat_card"]).exists()

    def test_auto_frame_interval_uses_fps(self, tiny_vod, tracking_uri, tmp_path):
        """No frame_interval given: derive one from the VOD's fps (~1 frame/sec)."""
        result = infer_new_vod(
            tiny_vod,
            output_dir=str(tmp_path / "report"),
            experiment_name="test_gold_usage",
        )
        assert result["n_frames"] == 3  # 90 frames @ 30fps, interval 30

    def test_rejects_missing_vod(self, tracking_uri, tmp_path):
        with pytest.raises(FileNotFoundError):
            infer_new_vod(
                str(tmp_path / "missing.mp4"),
                output_dir=str(tmp_path / "report"),
                experiment_name="test_gold_usage",
            )


class TestGoldDatasetRoundTrip:
    def test_synthetic_dataset_still_loads(self, tmp_path):
        ds = generate_synthetic_dataset(str(tmp_path / "ds"), num_samples=8, seed=1)
        with open(Path(ds) / "labels.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 8
