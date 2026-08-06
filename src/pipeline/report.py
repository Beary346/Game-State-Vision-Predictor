"""Gold reporting: quantified match timeline, score, and per-VOD MLflow runs.

AGENTS.md promises a match report your eye can't compute: an event timeline,
headline hit/whiff/round stats, a 0-100 score, and -- for every processed VOD
-- an MLflow run carrying params (VOD file + event thresholds), metrics
(event counts, mean OCR confidence) and artifacts (timeline JSON + stat card).

This module owns that reporting contract. The event objects come from
``events_gold.detect_events``, the stat card image from
``renderer_reportcard.render_stat_card``.
"""

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import mlflow

# Scoring weights (AGENTS.md's dopamine-hit formula). Round outcomes dominate;
# punishes reward clean damage; whiffs/being hit cost points. Base of 50.
SCORE_WEIGHTS: dict[str, float] = {
    "hit_landed": 2.0,
    "hit_taken": -2.0,
    "punish": 3.0,
    "whiff": -1.0,
    "round_win": 10.0,
    "round_loss": -12.0,
    "domain_alert": 0.0,
}
SCORE_BASE = 50.0

# Canonical weight keys (same names as the event types).
SCORE_WEIGHTS: dict[str, float] = {
    "hit_landed": 2.0,
    "hit_taken": -2.0,
    "punish": 3.0,
    "whiff": -1.0,
    "round_win": 10.0,
    "round_loss": -12.0,
    "domain_alert": 0.0,
}

# ``build_headline`` uses plural display names; the score weights use the event
# names. This alias lets ``compute_score`` accept either shape.
_WEIGHT_ALIAS: dict[str, str] = {
    "hit_landed": "hit_landed",
    "hits_landed": "hit_landed",
    "hit_taken": "hit_taken",
    "hits_taken": "hit_taken",
    "punish": "punish",
    "punishes": "punish",
    "whiff": "whiff",
    "whiffs": "whiff",
    "round_win": "round_win",
    "round_wins": "round_win",
    "round_loss": "round_loss",
    "round_losses": "round_loss",
    "domain_alert": "domain_alert",
    "domain_alerts": "domain_alert",
}


def ensure_tracking_uri() -> str:
    """Return an MLflow tracking URI that is safe to use from anywhere.

    Honors ``MLFLOW_TRACKING_URI`` (used by the tests and the web app); falls
    back to a SQLite DB under the system temp dir so spaces in project paths
    never trip it up. Every Gold stage shares one store.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    safe_path = os.path.join(tempfile.gettempdir(), "mlruns", "mlflow.db")
    os.makedirs(os.path.dirname(safe_path), exist_ok=True)
    uri = f"sqlite:///{safe_path}"
    mlflow.set_tracking_uri(uri)
    return uri


def mean_ocr_confidence(states: list) -> float:
    """Mean OCR confidence over the state tuples (0.0 for an empty stream).

    States without an ``ocr_confidence`` field (e.g. raw feature dicts) count
    as fully confident.
    """
    if not states:
        return 0.0
    confs = [
        float(s.get("ocr_confidence", 1.0))
        if isinstance(s, dict)
        else float(getattr(s, "ocr_confidence", 1.0))
        for s in states
    ]
    return float(sum(confs) / len(confs))


def build_headline(events: list) -> dict:
    """Count every event type into the plain-English headline stats.

    Also sums ``damage_dealt`` / ``damage_taken`` from the hit event details,
    which powers the per-round damage breakdown in the report.
    """
    counts = {
        "hits_landed": 0,
        "hits_taken": 0,
        "punishes": 0,
        "whiffs": 0,
        "round_wins": 0,
        "round_losses": 0,
        "domain_alerts": 0,
    }
    damage_dealt = 0.0
    damage_taken = 0.0
    for event in events:
        key = {
            "hit_landed": "hits_landed",
            "hit_taken": "hits_taken",
            "punish": "punishes",
            "whiff": "whiffs",
            "round_win": "round_wins",
            "round_loss": "round_losses",
            "domain_alert": "domain_alerts",
        }.get(event.type)
        if key:
            counts[key] += 1
        if event.type == "hit_landed":
            damage_dealt += float(event.detail.get("damage", 0.0))
        elif event.type == "hit_taken":
            damage_taken += float(event.detail.get("damage", 0.0))
    return {
        **counts,
        "total_events": len(events),
        "damage_dealt": round(damage_dealt, 4),
        "damage_taken": round(damage_taken, 4),
    }


def compute_score(headline_or_events, *, weights: dict | None = None, base: float = SCORE_BASE) -> float:
    """The match score in [0, 100] from a headline dict or raw event list."""
    if isinstance(headline_or_events, (list, tuple)):
        headline = build_headline(headline_or_events)
    else:
        headline = headline_or_events
    w = weights if weights is not None else SCORE_WEIGHTS
    score = base
    for key, value in headline.items():
        canonical = _WEIGHT_ALIAS.get(key)
        if canonical is not None:
            score += float(value) * w.get(canonical, 0.0)
    return float(max(0.0, min(100.0, round(score, 1))))


def _mean_enemy_health(state) -> float:
    """Mean enemy health on a frame (0.0 when none are detected)."""
    enemies = state.get("enemies", []) if isinstance(state, dict) else getattr(state, "enemies", [])
    healths = [
        float(e) if isinstance(e, (int, float)) else float(e.get("health", 0.0))
        for e in enemies
    ]
    return float(sum(healths) / len(healths)) if healths else 0.0


def build_timeline(
    *,
    states: list,
    events: list,
    vod_name: str = "",
    score: float | None = None,
    headline: dict | None = None,
    mean_ocr_confidence_: float | None = None,
    event_kwargs: dict | None = None,
) -> dict:
    """Assemble the full timeline JSON for one match.

    Structure::

        {
          "vod_file": "...", "n_frames": N, "score": s,
          "mean_ocr_confidence": c, "event_kwargs": {...},
          "headline": {...},
          "rounds": [ {...}, ... ],     # per-round damage + event counts
          "events": [ {...}, ... ],     # serialized gold events
          "state_curve": [ {...}, ... ] # per-frame health/label series
        }
    """
    hl = headline if headline is not None else build_headline(events)
    tl_score = score if score is not None else compute_score(hl)
    conf = mean_ocr_confidence_ if mean_ocr_confidence_ is not None else mean_ocr_confidence(states)

    rounds: dict[int, dict] = {}
    for state in states:
        r = int(state["round_index"]) if isinstance(state, dict) else int(getattr(state, "round_index", 0))
        ts = float(state["timestamp_sec"]) if isinstance(state, dict) else float(getattr(state, "timestamp_sec", 0.0))
        bucket = rounds.setdefault(
            r,
            {
                "round_index": r,
                "num_frames": 0,
                "start_sec": ts,
                "end_sec": ts,
                "damage_dealt": 0.0,
                "damage_taken": 0.0,
                "event_counts": {},
            },
        )
        bucket["num_frames"] += 1
        bucket["start_sec"] = min(bucket["start_sec"], ts)
        bucket["end_sec"] = max(bucket["end_sec"], ts)

    for event in events:
        bucket = rounds.get(event.round_index)
        if bucket is None:
            continue
        if event.type == "hit_landed":
            bucket["damage_dealt"] += float(event.detail.get("damage", 0.0))
        elif event.type == "hit_taken":
            bucket["damage_taken"] += float(event.detail.get("damage", 0.0))
        bucket["event_counts"][event.type] = bucket["event_counts"].get(event.type, 0) + 1

    state_curve = []
    for i, state in enumerate(states):
        frame = int(state["frame_index"]) if isinstance(state, dict) else int(getattr(state, "frame_index", i))
        ts = float(state["timestamp_sec"]) if isinstance(state, dict) else float(getattr(state, "timestamp_sec", 0.0))
        player = float(state["player_health"]) if isinstance(state, dict) else float(getattr(state, "player_health", 0.0))
        enemy_mean = _mean_enemy_health(state)
        ratio = player / (enemy_mean + 1e-12)
        state_curve.append(
            {
                "frame_index": frame,
                "timestamp_sec": round(ts, 4),
                "round_index": int(state["round_index"]) if isinstance(state, dict) else int(getattr(state, "round_index", 0)),
                "player_health": round(player, 4),
                "mean_enemy_health": round(enemy_mean, 4),
                "health_ratio": round(ratio, 4),
                "state_label": state.get("state_label") if isinstance(state, dict) else getattr(state, "state_label", None),
            }
        )

    return {
        "vod_file": vod_name,
        "n_frames": len(states),
        "score": float(tl_score),
        "mean_ocr_confidence": round(conf, 4),
        "event_kwargs": dict(event_kwargs or {}),
        "headline": hl,
        "rounds": [rounds[r] for r in sorted(rounds)],
        "events": [e.to_dict() for e in events],
        "state_curve": state_curve,
    }


def generate_report(
    *,
    states: list,
    events: list,
    output_dir: str,
    vod_name: str = "",
    score: float | None = None,
    headline: dict | None = None,
    mean_ocr_confidence_: float | None = None,
    event_kwargs: dict | None = None,
) -> dict:
    """Write the timeline JSON + stat card PNG for one match.

    Returns the report summary consumed by ``log_vod_report`` and by the CLI /
    notebook: artifact paths, the score, and the headline stats.
    """
    from src.pipeline.renderer_reportcard import render_stat_card

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timeline = build_timeline(
        states=states,
        events=events,
        vod_name=vod_name,
        score=score,
        headline=headline,
        mean_ocr_confidence_=mean_ocr_confidence_,
        event_kwargs=event_kwargs,
    )
    timeline_path = out / "timeline.json"
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2)

    card_path = render_stat_card(
        states=states,
        events=events,
        output_path=str(out / "stat_card.png"),
        vod_name=vod_name,
        score=timeline["score"],
        headline=timeline["headline"],
    )
    return {
        "vod_file": vod_name,
        "timeline_json": str(timeline_path),
        "stat_card": str(card_path),
        "score": timeline["score"],
        "headline": timeline["headline"],
        "timeline": timeline,
    }


def log_vod_report(
    *,
    states: list,
    events: list,
    output_dir: str,
    vod_name: str,
    experiment_name: str = "gold_vod_report",
    run_name: str | None = None,
    event_kwargs: dict | None = None,
) -> dict:
    """Run one match through report + stat card and track it as an MLflow run.

    AGENTS.md contract for a processed VOD:
      params    -> VOD file, n_frames and the event thresholds used
      metrics   -> event counts, mean OCR confidence, score, n_frames
      artifacts -> timeline.json + stat_card.png

    Returns the report summary plus the MLflow ``run_id``.
    """
    report = generate_report(
        states=states,
        events=events,
        output_dir=output_dir,
        vod_name=vod_name,
        event_kwargs=event_kwargs,
    )
    headline = report["headline"]

    ensure_tracking_uri()
    mlflow.set_experiment(experiment_name)
    if run_name is None:
        run_name = f"vod_{Path(vod_name).stem or 'match'}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        mlflow.log_params({"vod_file": vod_name, "n_frames": len(states)})
        if event_kwargs:
            mlflow.log_params({k: v for k, v in event_kwargs.items()})
        mlflow.log_metrics(
            {
                "score": float(report["score"]),
                "n_frames": float(len(states)),
                "total_events": float(headline["total_events"]),
                "mean_ocr_confidence": mean_ocr_confidence(states),
                "hits_landed": float(headline["hits_landed"]),
                "hits_taken": float(headline["hits_taken"]),
                "punishes": float(headline["punishes"]),
                "whiffs": float(headline["whiffs"]),
                "round_wins": float(headline["round_wins"]),
                "round_losses": float(headline["round_losses"]),
                "domain_alerts": float(headline["domain_alerts"]),
                "damage_dealt": float(headline["damage_dealt"]),
                "damage_taken": float(headline["damage_taken"]),
            }
        )
        mlflow.log_artifact(report["timeline_json"], artifact_path="")
        mlflow.log_artifact(report["stat_card"], artifact_path="")
        mlflow.log_text(json.dumps(report["timeline"], indent=2), "timeline_pretty.txt")

    report["run_id"] = run_id
    report["experiment_name"] = experiment_name
    return report


__all__ = [
    "SCORE_BASE",
    "SCORE_WEIGHTS",
    "build_headline",
    "build_timeline",
    "compute_score",
    "ensure_tracking_uri",
    "generate_report",
    "log_vod_report",
    "mean_ocr_confidence",
]