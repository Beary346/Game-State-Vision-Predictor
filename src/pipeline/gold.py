"""Gold layer: state classification + match reporting (winning / losing / stalemate).

The Gold layer owns everything downstream of Silver's per-frame state tuples:

1. **Training** -- ``train_gold_model`` / ``train_and_compare`` race a wide
   variety of classifiers (sklearn, XGBoost, PyTorch MLP) under MLflow on
   labeled Silver data; every run logs params, metrics, and confusion-matrix
   artifacts. The best model becomes the state-reader.
2. **Per-frame state classification** -- ``classify_states`` turns a stream of
   Silver tuples into per-frame ``winning`` / ``losing`` / ``stalemate``
   labels using a trained model (or the deterministic ``rule_based_label``).
3. **Event layer** -- ``detect_events`` (events_gold.py) turns state deltas
   into hits, punishes, whiffs, round outcomes, and domain alerts.
4. **Reporting** -- ``process_vod_report`` runs one match end-to-end: classify
   -> events -> timeline JSON + PIL stat card -> a per-VOD MLflow run with
   params (VOD file, event thresholds), metrics (event counts, mean OCR
   confidence, score) and artifacts (timeline JSON + stat card).
5. **Registry** -- ``model_registry.py`` registers the state-reader as a
   versioned MLflow model (Staging -> Production).

Feature vector layout (see ``FEATURE_NAMES``):
    player_health, mean_enemy_health, min_enemy_health, health_ratio,
    num_enemies, attacking, defending, damage_indicator

Labels are the player's *local initiative*, not health advantage:
    - winning    the player is currently striking the enemy (``attacking``)
    - losing     the player is currently being hit (``damage_indicator``)
    - stalemate  nothing is happening -- neutral, defending, no exchange

Health stays in the feature vector but is deliberately *not* part of the label:
a player on 1% HP who lands a strike is "winning" and one at 90% HP being
comboed is "losing". The event layer separately captures the health narrative
(hits landed, punishes, round outcomes).
"""

import argparse
import csv
import json
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from xgboost import XGBClassifier

from src.config import OUTPUTS_DIR
from src.pipeline.events_gold import detect_events
from src.pipeline.report import ensure_tracking_uri, log_vod_report
from src.pipeline.silver import H, W

# The six plain-English game states produced by the Gold layer.
STATE_LABELS: tuple[str, ...] = (
    "winning",
    "losing",
    "stalemate",
    "searching",
    "won",
    "lost",
)

# Order must match the vector produced by features_to_vector(). The base
# features carry the Silver HUD reads; the ``*_x_*`` names are NVIDIA-style
# feature crosses (combinations that carry more signal than either part alone).
FEATURE_NAMES: tuple[str, ...] = (
    "player_health",
    "mean_enemy_health",
    "min_enemy_health",
    "health_ratio",
    "num_enemies",
    "attacking",
    "defending",
    "damage_indicator",
    # Added: sprite geometry + per-ability cooldown states.
    "player_position_x",
    "player_position_y",
    "cooldown_0",
    "cooldown_1",
    "cooldown_2",
    "cooldown_3",
    "domain_ready",
    # Added: best-effort sprite/domain states (fixable in the labeler).
    "player_ragdoll",
    "enemy_ragdoll",
    "player_ultimate",
    "enemy_ultimate",
    "domain_active",
    "domain_activating",
    # NVIDIA-style feature crosses (combination features).
    "player_health_x_health_ratio",
    "player_health_x_player_position_x",
    "attacking_x_health_ratio",
    "defending_x_health_ratio",
    "player_ragdoll_x_player_health",
    "enemy_ragdoll_x_health_ratio",
    "attacking_x_num_enemies",
    "domain_active_x_attacking",
)

# How context fields (human observations) map onto Silver feature names so a
# labeler-confirmed value can correct a misread feature at training time.
CONTEXT_FEATURE_MAP: dict[str, str] = {
    "player_ragdolled": "player_ragdoll",
    "enemy_ragdolled": "enemy_ragdoll",
    "player_ultimate_active": "player_ultimate",
    "enemy_ultimate_active": "enemy_ultimate",
    "domain_active": "domain_active",
}

# Context fields that inform labeling but are not themselves model features.
CONTEXT_NOTES: tuple[str, ...] = (
    "enemy_visible",
    "player_moving",
    "round_start",
    "round_end",
)

_EPS = 1e-12


def _ensure_tracking_uri():
    """Alias of the shared tracker used by the training entry points."""
    return ensure_tracking_uri()


# ── Feature engineering ──────────────────────────────────────────────────────


def compute_health_ratio(silver_features: dict) -> float:
    """Player health divided by the mean enemy health (high = player winning)."""
    enemies = silver_features.get("enemies", [])
    enemy_healths = [float(e["health"]) for e in enemies]
    mean_enemy_health = float(np.mean(enemy_healths)) if enemy_healths else 0.0
    return float(silver_features["player_health"]) / (mean_enemy_health + _EPS)


def features_to_vector(silver_features: dict) -> np.ndarray:
    """Convert a SilverFeatures dict into the fixed Gold float32 vector.

    The vector layout must match ``FEATURE_NAMES`` exactly. Base reads default
    safely on old feature dicts; the trailing ``*_x_*`` entries are feature
    crosses (NVIDIA-style combination features) computed here so the model sees
    e.g. ``player_health * health_ratio`` as one input.
    """
    enemies = silver_features.get("enemies", [])
    enemy_healths = [float(e["health"]) for e in enemies]
    mean_enemy_health = float(np.mean(enemy_healths)) if enemy_healths else 0.0
    min_enemy_health = float(np.min(enemy_healths)) if enemy_healths else 0.0
    health = float(silver_features.get("player_health", 0.0))
    # Compute the ratio in float64 (Python floats) so the cast to float32
    # below only rounds once instead of accumulating float32 division error.
    health_ratio = health / (mean_enemy_health + _EPS)

    cooldowns = list(silver_features.get("ability_cooldowns", []))
    if len(cooldowns) < 4:
        # Old feature dicts had no per-slot cooldowns; fall back to the
        # single domain_ready boolean they carried.
        domain_ready = bool(silver_features.get("domain_ready", False))
        cooldowns = [float(domain_ready)] * 4

    pos = silver_features.get("player_position", (0.5, 0.5))
    if isinstance(pos, dict):
        pos = (pos.get("x", 0.5), pos.get("y", 0.5))
    elif len(pos) >= 2:
        pos = (float(pos[0]), float(pos[1]))
    else:
        pos = (0.5, 0.5)

    attacking = float(bool(silver_features.get("attacking", False)))
    defending = float(bool(silver_features.get("defending", False)))
    damage = float(bool(silver_features.get("damage_indicator", False)))
    player_ragdoll = float(bool(silver_features.get("player_ragdoll", False)))
    enemy_ragdoll = float(bool(silver_features.get("enemy_ragdoll", False)))
    domain_active = float(bool(silver_features.get("domain_active", False)))
    num_enemies = float(silver_features.get("num_enemies", len(enemies)))

    return np.array(
        [
            health,
            mean_enemy_health,
            min_enemy_health,
            health_ratio,
            num_enemies,
            attacking,
            defending,
            damage,
            pos[0],
            pos[1],
            float(cooldowns[0]),
            float(cooldowns[1]),
            float(cooldowns[2]),
            float(cooldowns[3]),
            float(bool(silver_features.get("domain_ready", all(c >= 0.95 for c in cooldowns)))),
            player_ragdoll,
            enemy_ragdoll,
            float(bool(silver_features.get("player_ultimate", False))),
            float(bool(silver_features.get("enemy_ultimate", False))),
            domain_active,
            float(bool(silver_features.get("domain_activating", False))),
            # ── Feature crosses ──
            health * health_ratio,
            health * pos[0],
            attacking * health_ratio,
            defending * health_ratio,
            player_ragdoll * health,
            enemy_ragdoll * health_ratio,
            attacking * num_enemies,
            domain_active * attacking,
        ],
        dtype=np.float32,
    )


# Health at or below this fraction counts as "dead" for the won/lost rules.
DEATH_HEALTH_THRESHOLD = 1e-3


def rule_based_label(silver_features: dict) -> str:
    """Deterministic initiative-based state label (matches the human labeler).

    Six states, decided in priority order — the first match wins:

    1. ``lost``      player health hit zero (death moment frames)
    2. ``searching`` no opponent is on the frame (no enemies detected)
    3. ``won``       enemy health hit zero (kill moment frames; needs an enemy)
    4. ``losing``    player is getting hit (damage flash)
    5. ``winning``   player is landing hits (attacking)
    6. ``stalemate`` nothing is happening (neutral / defending)

    Health is deliberately ignored for the three fight states (a player on 1%
    HP landing a strike is still "winning" — the event layer owns the health
    narrative), but defines the terminal ``won`` / ``lost`` death moments.
    """
    player_health = float(silver_features.get("player_health", 0.0))
    enemies = silver_features.get("enemies", [])
    if not enemies:
        min_enemy_health = 0.0
    else:
        enemy_healths = [float(e["health"]) for e in enemies]
        min_enemy_health = float(np.min(enemy_healths))
    damage = bool(silver_features.get("damage_indicator", False))
    attacking = bool(silver_features.get("attacking", False))

    if player_health <= DEATH_HEALTH_THRESHOLD:
        return "lost"
    if not enemies:
        return "searching"
    if min_enemy_health <= DEATH_HEALTH_THRESHOLD:
        return "won"
    if damage:
        return "losing"
    if attacking:
        return "winning"
    return "stalemate"


def apply_corrections(features: dict, scaffold: dict | None) -> dict:
    """Apply labeler corrections (silver_override + context) to a feature dict.

    The labeling tool can (a) override any Silver value directly and (b) record
    *context* observations about the frame. Both are applied on top of the
    feature dict here, so training (and rule recomputation) sees the corrected
    values instead of the raw (possibly misread) Silver reads.
    """
    if not scaffold:
        return features
    corrected = dict(features)
    override = scaffold.get("silver_override") or {}
    corrected.update(override)
    if "enemy0_hp" in override:
        # Convenience override for the first enemy's health bar read; rewrite
        # the enemies list so rules and feature mining consume the fix too.
        enemies = corrected.get("enemies") or []
        if enemies:
            enemies[0] = dict(enemies[0], health=float(override["enemy0_hp"]))
            corrected["enemies"] = enemies
    context = scaffold.get("context") or {}
    for context_key, feature_key in CONTEXT_FEATURE_MAP.items():
        if context_key in context and context[context_key] is not None:
            corrected[feature_key] = bool(context[context_key])
    return corrected


def _sample_features_for_label(rng: np.random.Generator, label: str, idx: int) -> dict:
    """Draw random SilverFeatures values that rule_based_label maps to *label*.

    Since the label rules are initiative-based, the synthesized flags are set
    so ``rule_based_label`` reproduces *label*:

    - winning = attacking (enemy present),  losing = damage flash,
    - stalemate = neither flag,            searching = no enemies at all,
    - won = every enemy at zero health,    lost = player health zero.

    Health and enemy values stay randomized — they are informative features
    but do not drive the synthetic label except through the death moments.
    """
    num_enemies = int(rng.integers(1, 3))
    position = (float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.9)))
    cooldown_fills = [float(rng.uniform(0.0, 1.0)) for _ in range(4)]

    if label == "searching":
        num_enemies = 0
    elif label == "won":
        num_enemies = 1  # the single enemy just died (kill moment)

    enemy_healths: list[float] = []
    for _ in range(num_enemies):
        if label == "won":
            health = 0.0
        else:
            health = float(rng.uniform(0.1, 0.95))
        enemy_healths.append(health)

    player_health = 0.0 if label == "lost" else float(rng.uniform(0.05, 0.95))

    # Flag matrix: damage has priority, attacking comes second, and the
    # terminal states force their health readings instead of the flags.
    if label == "winning":
        attacking, defending, damage_indicator = True, False, False
    elif label == "losing":
        attacking, defending, damage_indicator = False, bool(rng.integers(0, 2)), True
    elif label in ("searching", "won", "lost"):
        attacking, defending, damage_indicator = False, False, False
    else:
        attacking, defending, damage_indicator = False, bool(rng.integers(0, 2)), False

    enemies: list[dict] = []
    for health in enemy_healths:
        cx = int(rng.integers(100, W - 100))
        cy = int(rng.integers(100, H - 100))
        enemies.append(
            {
                "bbox": [cx - 30, cy - 30, 60, 60],
                "health": health,
                "health_bar_bbox": [cx - 25, cy - 40, 50, 8],
                "confidence": 1.0,
            }
        )

    return {
        "image_path": f"/synthetic/frame_{idx:04d}.png",
        "player_health": player_health,
        "player_position": [position[0], position[1]],
        "enemies": enemies,
        "attacking": attacking,
        "defending": defending,
        "damage_indicator": damage_indicator,
        "num_enemies": num_enemies,
        "ability_cooldowns": cooldown_fills,
        "domain_ready": bool(all(f >= 0.95 for f in cooldown_fills)),
        "player_ragdoll": bool(rng.integers(0, 2)) if label == "losing" else False,
        "enemy_ragdoll": False,
        "player_ultimate": False,
        "enemy_ultimate": False,
        "domain_active": False,
        "domain_activating": False,
    }


def generate_synthetic_dataset(output_dir: str, num_samples: int = 100, seed: int = 0) -> str:
    """Create a labeled dataset under *output_dir* for pipeline testing.

    Layout (identical to what real data will use)::

        output_dir/
            silver/*.json    # SilverFeatures dicts + an embedded "label" key
            labels.csv       # stem,label rows (source of truth)

    Labels are stratified across the three states so every class appears,
    and are reproducible for a given seed.
    """
    out = Path(output_dir)
    silver_dir = out / "silver"
    silver_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for i in range(num_samples):
        label = STATE_LABELS[int(rng.integers(0, len(STATE_LABELS)))]
        features = _sample_features_for_label(rng, label, i)
        if rule_based_label(features) != label:
            raise RuntimeError(f"Internal error: sampled features for {label!r} failed the rule check")
        features["label"] = label

        with open(silver_dir / f"frame_{i:04d}.json", "w") as f:
            json.dump(features, f, indent=2)
        rows.append({"stem": f"frame_{i:04d}", "label": label})

    with (out / "labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "label"])
        writer.writeheader()
        writer.writerows(rows)

    return str(out)


# ── Dataset loading ──────────────────────────────────────────────────────────


def load_labeled_dataset(data_root: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load Silver feature JSONs plus labels from a dataset directory.

    Expected layout::

        data_root/
            silver/*.json    # SilverFeatures dicts (optionally with "label")
            labels.csv       # stem,label — takes precedence over embedded labels
            labeling/*.json  # optional scaffolds: exclude/override/context

    Returns ``(X, y, stems)`` where ``X`` is the (n, len(FEATURE_NAMES))
    feature matrix and ``y`` holds integer labels 0..5 (indices into
    STATE_LABELS).

    When *some* labels exist (in ``labels.csv`` or embedded), samples without a
    label are skipped quietly — this is how real partial/skipped datasets train
    on just the reviewed subset. It raises only when *no* labels exist at all.

    Scaffolds (from the labeling tool) are honoured when present: ``exclude``
    and ``skip``-marked frames are dropped from training, and labeler
    corrections (``silver_override`` + ``context``) are applied to the feature
    vector, so the model trains on the reviewed truth rather than misreads.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    silver_dir = root / "silver"
    labeling_dir = root / "labeling"
    csv_labels: dict[str, str] = {}
    csv_path = root / "labels.csv"
    if csv_path.exists():
        with csv_path.open() as f:
            csv_labels = {row["stem"]: row["label"] for row in csv.DictReader(f)}

    scaffolds: dict[str, dict] = {}
    if labeling_dir.exists():
        for path in sorted(labeling_dir.glob("*_labeling.json")):
            stem = path.stem.removesuffix("_labeling")
            with open(path) as f:
                scaffolds[stem] = json.load(f)

    samples: list[dict] = []
    dropped_excluded = 0
    missing: list[str] = []
    for json_path in sorted(silver_dir.glob("*.json")):
        with open(json_path) as f:
            data = json.load(f)
        if "player_health" not in data:
            continue  # not a SilverFeatures file (e.g. Bronze metadata JSONs)

        stem = json_path.stem.removesuffix("_silver")

        scaffold = scaffolds.get(stem)
        if scaffold and (scaffold.get("exclude") or scaffold.get("skip")):
            dropped_excluded += 1
            continue

        label = csv_labels.get(stem, data.get("label"))
        if label is None:
            missing.append(stem)
            continue
        if label not in STATE_LABELS:
            raise ValueError(f"Invalid label {label!r} for sample {stem!r} (expected one of {STATE_LABELS})")
        samples.append((apply_corrections(data, scaffold), stem, label))

    if not samples:
        if missing:
            raise ValueError(
                "no labeled samples found (unlabeled stems include "
                + ", ".join(missing[:5])
                + "): review + export labels.csv under the labeling tool"
            )
        raise ValueError(f"No labeled samples found under {silver_dir}")

    X = np.stack([features_to_vector(data) for data, _, _ in samples]).astype(np.float32)
    y = np.array([STATE_LABELS.index(label) for _, _, label in samples], dtype=np.int64)
    stems = [stem for _, stem, _ in samples]
    if dropped_excluded:
        print(f"load_labeled_dataset: dropped {dropped_excluded} excluded/skipped scaffolded frames")
    return X, y, stems


# ── PyTorch MLP wrapper ──────────────────────────────────────────────────────


class TorchMLPClassifier:
    """Small PyTorch MLP exposing a scikit-learn-compatible fit/predict API.

    Standardizes inputs with the training set's mean/std, then trains a
    3-layer MLP with Adam + cross-entropy. Intended to be interchangeable
    with the sklearn models in MODEL_ZOO.
    """

    def __init__(self, hidden_dim: int = 16, epochs: int = 200, learning_rate: float = 1e-2, batch_size: int = 16, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seed = seed
        self.classes_: np.ndarray | None = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        class_to_idx = {int(c): i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[int(v)] for v in y], dtype=np.int64)

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        X_norm = (X - self.mean_) / self.std_

        torch.manual_seed(self.seed)
        self.model_ = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, len(self.classes_)),
        )
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        x_t = torch.from_numpy(X_norm)
        y_t = torch.from_numpy(y_idx)
        generator = torch.Generator().manual_seed(self.seed)
        n = x_t.size(0)

        self.model_.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(n, generator=generator)
            for start in range(0, n, self.batch_size):
                idx = permutation[start : start + self.batch_size]
                logits = self.model_(x_t[idx])
                loss = loss_fn(logits, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.model_.eval()
        return self

    def predict(self, X) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[probs.argmax(axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X_norm = (X - self.mean_) / self.std_
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X_norm))
            return torch.softmax(logits, dim=1).numpy()


# ── Model zoo ────────────────────────────────────────────────────────────────


def _filter_params(defaults: dict, overrides: dict | None, allowed: set[str]) -> dict:
    """Merge *overrides* into *defaults*, keeping only keys in *allowed*.

    The Gold pipeline shares one model_params dict across all six models
    (e.g. {"n_estimators": 10, "epochs": 5}), so each model must silently
    ignore parameters it does not support.
    """
    cfg = dict(defaults)
    if overrides:
        for key, value in overrides.items():
            if key in allowed:
                cfg[key] = value
    return cfg


def _logistic_regression(seed: int, overrides: dict | None):
    cfg = _filter_params({"max_iter": 2000, "random_state": seed}, overrides, {"C", "max_iter", "class_weight"})
    return make_pipeline(StandardScaler(), LogisticRegression(**cfg))


def _random_forest(seed: int, overrides: dict | None):
    cfg = _filter_params({"n_estimators": 100, "random_state": seed}, overrides, {"n_estimators", "max_depth", "min_samples_leaf"})
    return RandomForestClassifier(**cfg)


def _gradient_boosting(seed: int, overrides: dict | None):
    cfg = _filter_params({"random_state": seed}, overrides, {"n_estimators", "max_depth", "learning_rate"})
    return GradientBoostingClassifier(**cfg)


def _xgboost(seed: int, overrides: dict | None):
    cfg = _filter_params({"n_estimators": 100, "eval_metric": "mlogloss", "random_state": seed}, overrides, {"n_estimators", "max_depth", "learning_rate", "subsample"})
    return XGBClassifier(**cfg)


def _svc(seed: int, overrides: dict | None):
    # SVC(probability=True) is deprecated in sklearn >= 1.9; CalibratedClassifierCV
    # provides predict_proba with the same contract.
    cfg = _filter_params({"C": 1.0}, overrides, {"C", "gamma", "kernel"})
    svc = SVC(random_state=seed, **cfg)
    return make_pipeline(StandardScaler(), CalibratedClassifierCV(svc, ensemble=False))


def _pytorch_mlp(seed: int, overrides: dict | None):
    cfg = _filter_params({"hidden_dim": 16, "epochs": 200, "learning_rate": 1e-2, "batch_size": 16, "seed": seed}, overrides, {"hidden_dim", "epochs", "learning_rate", "batch_size"})
    return TorchMLPClassifier(**cfg)


# Name → factory(seed, overrides). Every model must expose fit/predict/predict_proba.
MODEL_ZOO: dict[str, object] = {
    "logistic_regression": _logistic_regression,
    "random_forest": _random_forest,
    "gradient_boosting": _gradient_boosting,
    "xgboost": _xgboost,
    "svc": _svc,
    "pytorch_mlp": _pytorch_mlp,
}


def _build_model(model_name: str, seed: int, overrides: dict | None):
    if model_name not in MODEL_ZOO:
        raise ValueError(f"Unknown model {model_name!r}; choose from {sorted(MODEL_ZOO)}")
    return MODEL_ZOO[model_name](seed, overrides)


# ── Visualization helpers (logged as MLflow artifacts) ───────────────────────


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str, title: str):
    from matplotlib import pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(STATE_LABELS))))
    n = len(STATE_LABELS)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(n),
        yticks=range(n),
        xticklabels=STATE_LABELS,
        yticklabels=STATE_LABELS,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    for row in range(n):
        for col in range(n):
            bright = cm[row, col] > cm.max() / 2
            ax.text(col, row, cm[row, col], ha="center", va="center", color="white" if bright else "black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_feature_correlations(X: np.ndarray, save_path: str):
    from matplotlib import pyplot as plt

    corr = np.corrcoef(X.astype(np.float64).T)
    corr = np.nan_to_num(corr, nan=0.0)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(FEATURE_NAMES)
    ax.figure.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Gold feature correlation matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_tsne(X: np.ndarray, y: np.ndarray, save_path: str, seed: int = 42):
    from matplotlib import pyplot as plt
    from sklearn.manifold import TSNE

    perplexity = int(min(30, max(5, X.shape[0] // 5)))
    embedding = TSNE(n_components=2, perplexity=perplexity, random_state=seed).fit_transform(X)

    fig, ax = plt.subplots(figsize=(9, 7))
    for label_idx in range(len(STATE_LABELS)):
        mask = y == label_idx
        if mask.any():
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=60, alpha=0.85, label=STATE_LABELS[label_idx])
    ax.set_title("t-SNE embedding of Gold feature space (colored by state)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ── Training ─────────────────────────────────────────────────────────────────


def train_gold_model(
    data_root: str,
    model_name: str = "random_forest",
    experiment_name: str = "gold_classifier",
    run_name: str | None = None,
    val_split: float = 0.2,
    seed: int = 42,
    model_params: dict | None = None,
) -> dict:
    """Train one Gold classifier on a labeled dataset, fully tracked by MLflow.

    Parameters
    ----------
    data_root : str
        Dataset directory containing ``silver/*.json`` and ``labels.csv``.
    model_name : str
        One of MODEL_ZOO (e.g. "random_forest", "xgboost", "pytorch_mlp").
    experiment_name : str
        MLflow experiment name.
    run_name : str | None
        MLflow run name (auto-generated if ``None``).
    val_split : float
        Fraction of data held out for validation.
    seed : int
        Random seed for reproducibility.
    model_params : dict | None
        Optional hyperparameter overrides; each model accepts only the
        parameters it supports.

    Returns a dict with the run_id, accuracy, macro-F1, and weighted F1.
    """
    X, y, _ = load_labeled_dataset(data_root)
    model = _build_model(model_name, seed, model_params)

    # Stratify the split when class counts allow it; fall back to plain split.
    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=seed, stratify=y)
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=seed)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = float(accuracy_score(y_val, y_pred))
    macro_f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))

    _ensure_tracking_uri()
    mlflow.set_experiment(experiment_name)
    if run_name is None:
        run_name = f"gold_{model_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        mlflow.log_params(
            {
                "model_name": model_name,
                "seed": seed,
                "val_split": val_split,
                "n_samples": len(X),
                "n_train": len(X_train),
                "n_val": len(X_val),
                **{f"train_{label}": int((y_train == i).sum()) for i, label in enumerate(STATE_LABELS)},
            }
        )
        mlflow.log_metrics({"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1})

        with tempfile.TemporaryDirectory() as tmp:
            cm_path = str(Path(tmp) / "confusion_matrix.png")
            _plot_confusion_matrix(y_val, y_pred, cm_path, title=f"Confusion Matrix — {model_name}")
            mlflow.log_artifact(cm_path)

        report = classification_report(y_val, y_pred, labels=list(range(len(STATE_LABELS))), target_names=STATE_LABELS, zero_division=0, digits=3)
        mlflow.log_text(report, "classification_report.txt")

        # cloudpickle (instead of skops) so XGBoost and TorchMLPClassifier
        # round-trip without the skops "untrusted types" safety check.
        model_info = mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle")
        # MLflow 3.x stores logged models in an experiment-level model store and
        # returns the canonical model URI (models:/m-<id>); the classic
        # runs:/<run>/model path no longer resolves against that layout.
        model_path = model_info.model_uri
        mlflow.log_param("logged_model", model_path)

    print(f"{model_name:22s}  acc={accuracy:.3f}  macro_f1={macro_f1:.3f}  run={run_id}")
    return {
        "model_name": model_name,
        "run_id": run_id,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "model_path": model_path,
    }


def _fresh_experiment(name: str):
    """Reset an MLflow experiment so repeated comparisons start from a clean slate.

    MLflow refuses to reuse a deleted experiment name, so instead of
    delete/recreate we restore any leftover and mark all existing runs
    deleted (search_runs then ignores them).
    """
    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        if experiment.lifecycle_stage == "deleted":
            client.restore_experiment(experiment.experiment_id)
            experiment = mlflow.get_experiment(experiment.experiment_id)
        for run in mlflow.search_runs(experiment_names=[name], output_format="list"):
            client.delete_run(run.info.run_id)
    return mlflow.set_experiment(name)


def train_and_compare(
    data_root: str,
    experiment_name: str = "gold_classifier",
    val_split: float = 0.2,
    seed: int = 42,
    model_params: dict | None = None,
) -> pd.DataFrame:
    """Train every model in MODEL_ZOO under MLflow and return a ranked leaderboard.

    Logs one run per model (params, metrics, confusion matrix, classification
    report) plus a final "leaderboard" run containing the leaderboard CSV,
    a feature correlation matrix, and a t-SNE embedding of the feature space.
    """
    _ensure_tracking_uri()
    _fresh_experiment(experiment_name)

    results = []
    for name in MODEL_ZOO:
        result = train_gold_model(
            data_root,
            model_name=name,
            experiment_name=experiment_name,
            run_name=f"gold_{name}",
            val_split=val_split,
            seed=seed,
            model_params=model_params,
        )
        results.append(result)

    df = pd.DataFrame(results).sort_values("macro_f1", ascending=False).reset_index(drop=True)

    X, y, _ = load_labeled_dataset(data_root)
    with mlflow.start_run(run_name="leaderboard"):
        mlflow.log_params(
            {
                "n_models": len(df),
                "n_samples": len(X),
                "n_features": X.shape[1],
                "best_model": str(df.iloc[0]["model_name"]),
                "seed": seed,
            }
        )
        for _, row in df.iterrows():
            mlflow.log_metric(f"macro_f1_{row['model_name']}", float(row["macro_f1"]))

        with tempfile.TemporaryDirectory() as tmp:
            leaderboard_path = Path(tmp) / "leaderboard.csv"
            df.to_csv(leaderboard_path, index=False)
            mlflow.log_artifact(str(leaderboard_path))

            corr_path = Path(tmp) / "feature_correlations.png"
            _plot_feature_correlations(X, str(corr_path))
            mlflow.log_artifact(str(corr_path))

            tsne_path = Path(tmp) / "tsne_embedding.png"
            _plot_tsne(X, y, str(tsne_path), seed=seed)
            mlflow.log_artifact(str(tsne_path))

    return df


# ── Inference ────────────────────────────────────────────────────────────────


def predict(features, model) -> tuple[str, dict[str, float]]:
    """Predict the game state for Silver features (dict or vector) with a trained model.

    Returns ``(label, probabilities)`` where probabilities covers every state
    in STATE_LABELS (unseen classes are reported as 0.0).
    """
    if isinstance(features, np.ndarray):
        vector = features.astype(np.float32, copy=False)
    else:
        vector = features_to_vector(features)

    X = vector.reshape(1, -1)
    pred_idx = int(np.asarray(model.predict(X))[0])
    proba = np.asarray(model.predict_proba(X))[0]
    classes = [int(c) for c in model.classes_]

    probs = {label: 0.0 for label in STATE_LABELS}
    for class_idx, prob in zip(classes, proba):
        probs[STATE_LABELS[class_idx]] = float(prob)

    return STATE_LABELS[pred_idx], probs


# ── Train -> register -> use on unseen VODs ──────────────────────────────────


def register_best_model(
    leaderboard: pd.DataFrame,
    *,
    name: str = "state_reader",
    stage: str = "Production",
) -> dict:
    """Register the leaderboard winner as a versioned MLflow model.

    ``train_and_compare`` returns a leaderboard sorted by macro-F1 (best first);
    this takes the top row, registers its logged model under ``name`` (default
    ``state_reader``), stamps the training metrics as version tags, and moves it
    straight to the requested stage (Production by default) so inference on new
    VODs picks it up via ``load_state_reader``.

    Returns the model-registry version summary dict.
    """
    if leaderboard is None or leaderboard.empty:
        raise ValueError("empty leaderboard: nothing to register (train first?)")

    from src.pipeline.model_registry import register_state_reader

    winner = leaderboard.iloc[0]
    result = register_state_reader(
        winner["model_path"],
        name=name,
        stage=stage,
        tags={
            "model_name": str(winner["model_name"]),
            "accuracy": str(float(winner["accuracy"])),
            "macro_f1": str(float(winner["macro_f1"])),
            "weighted_f1": str(float(winner["weighted_f1"])),
            "run_id": str(winner.get("run_id", "")),
        },
    )
    print(
        f"Registered {name} v{result['version']} ({winner['model_name']}) "
        f"-> {stage} (run {result['run_id']})"
    )
    return result


def infer_new_vod(
    vod_file: str,
    *,
    model=None,
    frame_interval: int | None = None,
    output_dir: str | None = None,
    experiment_name: str = "gold_vod_report",
    run_name: str | None = None,
    event_kwargs: dict | None = None,
) -> dict:
    """Run a brand-new VOD through Bronze -> Silver -> Gold and return its MLflow report.

    The one-stop entry point for *outside data*: a VOD that was never part of
    the training set. It extracts frames (Bronze), reads the HUD features
    (Silver), classifies every frame with the supplied ``model`` (or the
    registered production ``state_reader``, or the deterministic rules when no
    model exists), then produces the timeline + stat card + per-VOD MLflow run.

    ``frame_interval`` defaults to the VOD's fps (≈1 sampled frame per second),
    matching the data-collection convention.
    """
    from src.ingestor_bronze import get_vod_info, process_vod
    from src.pipeline.silver import process_frames

    if not Path(vod_file).exists():
        raise FileNotFoundError(f"VOD not found: {vod_file}")

    vod_meta = get_vod_info(vod_file)
    interval = frame_interval if frame_interval is not None else round(vod_meta.fps)

    classifier = model
    if classifier is None:
        try:
            from src.pipeline.model_registry import load_state_reader

            classifier = load_state_reader()  # registered Production state-reader
        except RuntimeError:
            classifier = None  # no registered model yet -> rule-based fallback

    with tempfile.TemporaryDirectory(prefix="gsvp_infer_") as tmp:
        bronze_dir = str(Path(tmp) / "bronze")
        silver_dir = str(Path(tmp) / "silver")
        process_vod(vod_file, bronze_dir, frame_interval=interval)
        process_frames(bronze_dir, silver_dir)
        states = load_states_from_dir(silver_dir)

    return process_vod_report(
        states=states,
        model=classifier,
        vod_file=vod_file,
        output_dir=output_dir,
        experiment_name=experiment_name,
        run_name=run_name,
        event_kwargs=event_kwargs,
    )


# ── Per-stream classification & end-to-end VOD reports ───────────────────────


def classify_state(features, model=None) -> tuple[str, dict[str, float]]:
    """Classify one SilverFeatures dict into ``(label, probs)``.

    With a trained ``model`` the model prediction (plus full probabilities) is
    used; without one the deterministic ``rule_based_label`` fills in, so the
    pipeline stays usable before real training data exists.
    """
    state = features if isinstance(features, dict) else asdict(features)
    if model is None:
        return rule_based_label(state), {}
    return predict(state, model)


def classify_states(states: list, model=None) -> list[dict]:
    """Label every frame of a match stream; return one row per frame.

    Each row is ``{"state": <state + state_label>, "state_probs": {...},
    "frame_index": n}``. The ``state`` dicts keep every original key so the
    event layer and the report can still read healths, attacks, and domains.
    """
    out: list[dict] = []
    for i, features in enumerate(states):
        state = features if isinstance(features, dict) else asdict(features)
        # Normalize to the full shape the event/report layers expect. Minimal
        # 8-field feature dicts (older annotations, hand-built tests) get safe
        # defaults instead of crashing the report.
        if "frame_index" not in state:
            state["frame_index"] = int(state.get("timestamp_sec", 0.0) * 30 if state.get("timestamp_sec") else i)
        state.setdefault("round_index", 0)
        state.setdefault("timestamp_sec", float(state["frame_index"]) / 30.0)
        state.setdefault("clock_sec", None)
        state.setdefault("domain_ready", False)
        state.setdefault("ocr_confidence", 1.0)
        state.setdefault("attacking", False)
        state.setdefault("defending", False)
        state.setdefault("damage_indicator", False)
        state.setdefault("num_enemies", len(state.get("enemies", [])))
        label, probs = classify_state(state, model)
        state["state_label"] = label
        out.append(
            {
                "state": state,
                "state_probs": probs,
                "frame_index": state.get("frame_index", i),
            }
        )
    return out


def load_states_from_dir(states_dir: str) -> list[dict]:
    """Load every ``*_silver.json`` in a directory as one ordered frame stream.

    Skips non-state sidecars (e.g. Bronze metadata, rounds.json) by requiring
    the ``player_health`` key, then sorts by frame_index.
    """
    states_dir = Path(states_dir)
    states: list[dict] = []
    for path in sorted(states_dir.glob("*_silver.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "player_health" not in data:
            continue
        states.append(data)
    states.sort(key=lambda s: s.get("frame_index", 0))
    if not states:
        raise ValueError(f"no *_silver.json state files with a player_health key in {states_dir}")
    return states


def process_vod_report(
    *,
    states: list,
    model=None,
    vod_file: str = "",
    output_dir: str | None = None,
    experiment_name: str = "gold_vod_report",
    run_name: str | None = None,
    event_kwargs: dict | None = None,
) -> dict:
    """Run the full Gold pipeline over one match and return its MLflow report.

    classify -> events -> timeline JSON + PIL stat card -> per-VOD MLflow run.
    With ``output_dir=None`` the artifacts land under ``outputs/gold_<stem>``.

    Returns the ``log_vod_report`` summary (run_id, timeline_json, stat_card,
    score, headline...) extended with ``vod_file`` and ``n_frames``.
    """
    classified = classify_states(states, model=model)
    stream = [row["state"] for row in classified]

    events = detect_events(stream, **(event_kwargs or {}))

    stem = Path(vod_file).stem or "match"
    out_dir = output_dir or str(OUTPUTS_DIR / f"gold_{stem}")

    result = log_vod_report(
        states=stream,
        events=events,
        output_dir=out_dir,
        vod_name=vod_file,
        experiment_name=experiment_name,
        run_name=run_name,
        event_kwargs=event_kwargs,
    )
    result["vod_file"] = vod_file
    result["n_frames"] = len(stream)
    result["n_events"] = len(events)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gold: train the state-reader, and render per-VOD match reports")
    ap.add_argument("--data-root", default=None, help="path to labeled dataset (silver/ + labels.csv)")
    ap.add_argument("--synthetic", type=int, default=None, help="generate N synthetic labeled samples into a temp dir")
    ap.add_argument("--model", default=None, help="train only this model (default: all models in the zoo)")
    ap.add_argument("--experiment", default="gold_classifier", help="MLflow experiment name")
    ap.add_argument("--val-split", type=float, default=0.2, help="validation split fraction")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--vod-report", action="store_true", help="process a VOD's silver state JSONs into a match report")
    ap.add_argument("--states-dir", default=None, help="directory of *_silver.json state files (for --vod-report)")
    ap.add_argument("--states-model", default=None, help="'run:ID/model' URI of a trained state-reader (for --vod-report)")
    ap.add_argument("--infer-vod", default=None, help="raw VOD path to run end-to-end (Bronze -> Silver -> Gold report) on unseen footage")
    ap.add_argument("--infer-model", default=None, help="'run:ID/model' or 'models:/state_reader/1' URI for --infer-vod (default: registered production state-reader, else rules)")
    ap.add_argument("--infer-interval", type=int, default=None, help="frame interval for --infer-vod (default: VOD fps)")
    ap.add_argument("--register-best", action="store_true", help="register the leaderboard winner as the production state_reader after training")
    args = vars(ap.parse_args())

    if args["infer_vod"]:
        if not Path(args["infer_vod"]).exists():
            ap.error(f"--infer-vod path not found: {args['infer_vod']}")
        model = None
        if args["infer_model"]:
            model = mlflow.sklearn.load_model(args["infer_model"])
        result = infer_new_vod(
            args["infer_vod"],
            model=model,
            frame_interval=args["infer_interval"],
            experiment_name=args["experiment"],
            event_kwargs={},
        )
        print(f"\nScore {result['score']:.0f}/100 for {result['vod_file']}")
        print(
            f"  hits {result['headline']['hits_landed']}  punishes {result['headline']['punishes']}"
            f"  whiffs {result['headline']['whiffs']}  rounds {result['headline']['round_wins']}W "
            f"{result['headline']['round_losses']}L"
        )
        print(f"Timeline : {result['timeline_json']}")
        print(f"Stat card: {result['stat_card']}")
        print(f"MLflow run: {result['run_id']} ({result['experiment_name']})")
        raise SystemExit(0)

    if args["vod_report"]:
        if not args["states_dir"]:
            ap.error("--vod-report needs --states-dir")
        model = None
        if args["states_model"]:
            model = mlflow.sklearn.load_model(args["states_model"])
        states = load_states_from_dir(args["states_dir"])
        print(f"Reading {len(states)} frames from {args['states_dir']}...")
        result = process_vod_report(
            states=states,
            model=model,
            vod_file=args["states_dir"],
            experiment_name=args["experiment"],
            event_kwargs={},
        )
        print(f"\nScore {result['score']:.0f}/100 for {result['vod_file']}")
        print(
            f"  hits {result['headline']['hits_landed']}  punishes {result['headline']['punishes']}"
            f"  whiffs {result['headline']['whiffs']}  rounds {result['headline']['round_wins']}W "
            f"{result['headline']['round_losses']}L"
        )
        print(f"Timeline : {result['timeline_json']}")
        print(f"Stat card: {result['stat_card']}")
        print(f"MLflow run: {result['run_id']} ({result['experiment_name']})")
        raise SystemExit(0)

    data_root = args["data_root"]
    if data_root is None:
        if args["synthetic"] is None:
            ap.error("provide --data-root or --synthetic N")
        tmp = tempfile.mkdtemp(prefix="gold_synthetic_")
        print(f"Generating {args['synthetic']} synthetic samples in {tmp}...")
        data_root = generate_synthetic_dataset(tmp, num_samples=args["synthetic"], seed=args["seed"])

    if args["model"]:
        result = train_gold_model(
            data_root,
            model_name=args["model"],
            experiment_name=args["experiment"],
            val_split=args["val_split"],
            seed=args["seed"],
        )
        print(f"\nModel: {result['model_name']}  accuracy={result['accuracy']:.4f}  macro_f1={result['macro_f1']:.4f}")
    else:
        leaderboard = train_and_compare(
            data_root,
            experiment_name=args["experiment"],
            val_split=args["val_split"],
            seed=args["seed"],
        )
        print("\nLeaderboard (sorted by macro-F1):")
        print(leaderboard[["model_name", "accuracy", "macro_f1", "weighted_f1"]].to_string(index=False))
        print(f"\nBest model: {leaderboard.iloc[0]['model_name']}")
        if args["register_best"]:
            register_best_model(leaderboard, name="state_reader")
