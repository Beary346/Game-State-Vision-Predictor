"""Gold model registry: the state-reader as a versioned MLflow model.

AGENTS.md: the state-reader (the trained Gold classifier) must be registered
as a *versioned model* -- so every training round is kept, comparable, and
promotable (Staging -> Production) without losing history.

This module wraps the MLflow Model Registry with a tiny domain vocabulary:
the registered model is called ``state_reader``, versions carry the metrics
of the training run as tags, and stages map to the classic
Staging/Production lifecycle.
"""

import mlflow

# Registered-model name shared by every Gold training round. A "state reader"
# is exactly what the Gold classifier is: it reads a state tuple and answers
# winning / losing / stalemate.
REGISTERED_MODEL_NAME = "state_reader"

# Stage names kept constant so callers don't typo them into the registry.
STAGE_STAGING = "Staging"
STAGE_PRODUCTION = "Production"


def _client() -> mlflow.tracking.MlflowClient:
    """The registry client bound to the shared tracking store."""
    from src.pipeline.report import ensure_tracking_uri

    ensure_tracking_uri()
    return mlflow.tracking.MlflowClient()


def _registered_model(name: str):
    """The registered model or ``None`` -- MLflow 3.x raises when missing."""
    import mlflow.exceptions

    try:
        return _client().get_registered_model(name)
    except mlflow.exceptions.MlflowException:
        return None


def register_state_reader(
    model_uri: str,
    *,
    name: str = REGISTERED_MODEL_NAME,
    stage: str = STAGE_STAGING,
    tags: dict | None = None,
) -> dict:
    """Register a trained classifier as the next ``state_reader`` version.

    Parameters
    ----------
    model_uri
        The logged model to register -- typically ``ModelInfo.model_uri``
        (``models:/m-<model_id>``) straight out of ``train_gold_model``.
    name
        Registered-model name (defaults to the state-reader contract).
    stage
        Initial lifecycle stage of the new version.
    tags
        Free-form metadata stamped onto the version, e.g.
        ``{"accuracy": 0.94, "macro_f1": 0.92, "model_name": "xgboost"}``.

    Returns a dict describing the new version (name, version, stage, tags,
    run_id).
    """
    client = _client()
    if not _registered_model(name):
        client.create_registered_model(name)

    version = client.create_model_version(name, model_uri, run_id=None)
    version_number = int(version.version)

    for key, value in (tags or {}).items():
        client.set_model_version_tag(name, version_number, key, str(value))

    # New MLflow versions start with stage "None"; set the requested stage
    # explicitly (newer MLflow no longer auto-assigns "Staging").
    if stage:
        client.transition_model_version_stage(name, str(version_number), stage)

    return _version_summary(client.get_model_version(name, str(version_number)))


def list_state_reader_versions(*, name: str = REGISTERED_MODEL_NAME) -> list[dict]:
    """All versions of the state-reader, oldest first ([] if unregistered)."""
    if not _registered_model(name):
        return []
    versions = _client().search_model_versions(f"name = '{name}'")
    versions.sort(key=lambda v: int(v.version))
    return [_version_summary(v) for v in versions]


def transition_state_reader_version(
    model_version: int | str,
    stage: str = STAGE_PRODUCTION,
    *,
    name: str = REGISTERED_MODEL_NAME,
) -> dict:
    """Promote/demote one state-reader version to *stage* (e.g. Production)."""
    client = _client()
    client.transition_model_version_stage(name, str(model_version), stage)
    return _version_summary(client.get_model_version(name, str(model_version)))


def get_state_reader(
    *,
    name: str = REGISTERED_MODEL_NAME,
    model_version: int | str | None = None,
    stage: str | None = None,
) -> dict | None:
    """Resolve a state-reader version (None when nothing matches).

    Prefers an explicit ``model_version``; otherwise returns the newest
    version currently in ``stage`` (default: newest overall).
    """
    client = _client()
    if not _registered_model(name):
        return None

    if model_version is not None:
        try:
            version = client.get_model_version(name, str(model_version))
        except mlflow.exceptions.MlflowException:
            return None
        return _version_summary(version)

    versions = list_state_reader_versions(name=name)
    if not versions:
        return None
    if stage is None:
        return versions[-1]
    staged = [v for v in versions if v["stage"] == stage]
    return staged[-1] if staged else None


def load_state_reader(
    *,
    name: str = REGISTERED_MODEL_NAME,
    model_version: int | str | None = None,
    stage: str | None = None,
):
    """Load a registered state-reader as a callable classifier.

    Uses the classic ``models:/<name>/<version>`` URI so loading stays stable
    even as aliases/stages evolve. Raises when no version exists.
    """
    info = get_state_reader(name=name, model_version=model_version, stage=stage)
    if info is None:
        raise RuntimeError(
            f"No state-reader registered{'' if model_version is None else f' (version {model_version})'} "
            f"under {name!r} -- train and register one first."
        )
    return mlflow.sklearn.load_model(f"models:/{name}/{info['version']}")


def _version_summary(version) -> dict:
    """A stable, serializable view of one model version."""
    tags = dict(version.tags or {})
    return {
        "name": version.name,
        "version": int(version.version),
        "stage": str(getattr(version, "current_stage", "") or "None"),
        "run_id": getattr(version, "run_id", None),
        "tags": tags,
    }


__all__ = [
    "REGISTERED_MODEL_NAME",
    "STAGE_PRODUCTION",
    "STAGE_STAGING",
    "get_state_reader",
    "list_state_reader_versions",
    "load_state_reader",
    "register_state_reader",
    "transition_state_reader_version",
]