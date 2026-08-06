"""Tests for the Gold model registry: the trained state-reader as a versioned model.

AGENTS.md: the Gold layer's state-reader must be registered as a versioned
model (MLflow Model Registry), so every training round can be tracked,
compared, and promoted (Staging -> Production) without losing history.
"""

import os

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.pipeline.model_registry import (
    REGISTERED_MODEL_NAME,
    get_state_reader,
    list_state_reader_versions,
    load_state_reader,
    register_state_reader,
    transition_state_reader_version,
)


@pytest.fixture
def tracking_uri(tmp_path) -> str:
    uri = f"sqlite:///{tmp_path / 'registry.db'}"
    os.environ["MLFLOW_TRACKING_URI"] = uri
    os.environ["MLFLOW_REGISTRY_URI"] = uri
    yield uri
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    os.environ.pop("MLFLOW_REGISTRY_URI", None)


@pytest.fixture
def logged_model_uri(tracking_uri, tmp_path) -> str:
    """A tiny trained classifier logged to a fresh MLflow run."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, (40, 4))
    y = np.where(X[:, 1] > X[:, 0], 1, 0)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X, y)
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(model, "model")
    return info.model_uri


@pytest.fixture
def cleanup_registry(tracking_uri):
    yield
    client = mlflow.tracking.MlflowClient()
    for mv in client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'"):
        client.delete_model_version(REGISTERED_MODEL_NAME, mv.version)
    from mlflow.exceptions import MlflowException

    try:
        client.get_registered_model(REGISTERED_MODEL_NAME)
    except MlflowException:
        pass
    else:
        client.delete_registered_model(REGISTERED_MODEL_NAME)


class TestRegisterStateReader:
    def test_registers_a_version_with_tags(self, logged_model_uri, cleanup_registry):
        result = register_state_reader(
            logged_model_uri,
            tags={"accuracy": 0.95, "macro_f1": 0.93, "model_name": "logistic_regression"},
        )
        assert result["name"] == REGISTERED_MODEL_NAME
        assert result["version"] == 1
        assert result["stage"] == "Staging"
        assert result["tags"]["accuracy"] == "0.95"

    def test_second_registration_bumps_version(self, logged_model_uri, cleanup_registry):
        register_state_reader(logged_model_uri)
        second = register_state_reader(logged_model_uri)
        assert second["version"] == 2

    def test_custom_registered_name(self, logged_model_uri, tracking_uri):
        result = register_state_reader(logged_model_uri, name="custom_reader")
        assert result["name"] == "custom_reader"
        client = mlflow.tracking.MlflowClient()
        client.delete_registered_model("custom_reader")


class TestListVersions:
    def test_lists_all_versions(self, logged_model_uri, cleanup_registry):
        register_state_reader(logged_model_uri, tags={"model_name": "rf"})
        register_state_reader(logged_model_uri, tags={"model_name": "xgb"})
        versions = list_state_reader_versions()
        assert len(versions) == 2
        assert {v["version"] for v in versions} == {1, 2}
        tags = {v["version"]: v["tags"]["model_name"] for v in versions}
        assert tags == {1: "rf", 2: "xgb"}

    def test_empty_when_unregistered(self, tracking_uri):
        assert list_state_reader_versions(name="does_not_exist") == []


class TestPromotion:
    def test_transition_to_production(self, logged_model_uri, cleanup_registry):
        result = register_state_reader(logged_model_uri)
        promoted = transition_state_reader_version(result["version"], stage="Production")
        assert promoted["stage"] == "Production"

    def test_get_returns_latest_for_stage(self, logged_model_uri, cleanup_registry):
        register_state_reader(logged_model_uri)
        register_state_reader(logged_model_uri)
        transition_state_reader_version(2, stage="Production")
        current = get_state_reader(stage="Production")
        assert current["version"] == 2
        assert current["stage"] == "Production"


class TestLoad:
    def test_load_works_for_production(self, logged_model_uri, cleanup_registry):
        result = register_state_reader(logged_model_uri)
        transition_state_reader_version(result["version"], stage="Production")
        model = load_state_reader()
        # X[:, 1] > X[:, 0] labels the split classes; validate on a known point.
        pred = model.predict(np.array([[0.2, 0.9, 0.5, 0.5]]))
        assert np.asarray(pred).shape == (1,)
        assert np.asarray(pred)[0] in (0, 1)

    def test_load_by_model_version(self, logged_model_uri, cleanup_registry):
        result = register_state_reader(logged_model_uri)
        model = load_state_reader(model_version=result["version"])
        assert hasattr(model, "predict")

    def test_nothing_registered_returns_none(self, tracking_uri):
        assert get_state_reader() is None