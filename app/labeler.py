"""FastAPI label-review web tool for Gold Silver frames.

Run from the project root::

    uvicorn app.labeler:app --host 127.0.0.1 --port 8765
    # open http://127.0.0.1:8765

Workflow:
    1. ``python -m src.pipeline.labeling --init``  (creates the empty scaffolds,
       already done for the current corpus)
    2. Start the server and open the URL.
    3. Review frames: see the bronze frame, its Silver features and the
       rule-bootstrap label, then click one of the six states
       ``winning / losing / stalemate / searching / won / lost`` (or ``Skip``
       for out-of-distribution noise, ``Exclude`` to remove the frame from the
       dataset). Every answer is written straight back into
       ``data/gold/labeling/<stem>_labeling.json``.
    4. Where Silver misread a frame, fix it directly: check the structured
       ``context`` observations (e.g. ``enemy_visible``, ``player_ragdolled``)
       and/or key in ``silver_override`` values (e.g. ``player_health``) — the
       rule label refreshes instantly and training consumes the corrections.
    5. Hit ``Export labels.csv`` once done (or ``/api/export``) -- this writes
       ``data/gold/labels.csv`` from every labeled, non-skipped, non-excluded
       scaffold.
    6. Train Gold: ``python -m src.pipeline.gold --data-root data/gold``

Keyboard shortcuts (from the review page):
    ``1/2/3/4/5/6``  label winning/losing/stalemate/searching/won/lost
    ``s``      skip / out-of-distribution
    ``x``      exclude the frame from the dataset
    ``u``      undo the current answer
    ``n``/``p`` next / previous frame
    ``e``      export labels.csv
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import PROJECT_ROOT
from src.pipeline import labeling
from src.pipeline.gold import apply_corrections, rule_based_label

app = FastAPI(title="Game State Vision Predictor — Label Review")

STATIC_DIR = Path(__file__).parent / "static"
LABELING_DIR = PROJECT_ROOT / "data" / "gold" / "labeling"
SILVER_DIR = PROJECT_ROOT / "data" / "gold" / "silver"
LABELS_CSV = PROJECT_ROOT / "data" / "gold" / "labels.csv"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _scaffold_for(stem: str) -> dict:
    try:
        return labeling.load_scaffold(LABELING_DIR, stem)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"no labeling file for stem {stem!r}") from None


def _image_path_for(stem: str) -> Path:
    """Resolve the frame image on disk from a labeling scaffold."""
    payload = _scaffold_for(stem)
    candidate = Path(str(payload["image_path"]))
    candidate = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    if not candidate.exists():
        raise HTTPException(status_code=404, detail=f"frame image missing for stem {stem!r}")
    return candidate


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/summary")
def api_summary():
    return labeling.summary(LABELING_DIR)


@app.get("/api/states")
def api_states():
    """Compact label map for every stem — powers the sidebar row colours."""
    items = {}
    for stem, payload in labeling.iter_scaffolds(LABELING_DIR, load=True):
        items[stem] = {
            "label": payload.get("label"),
            "skip": bool(payload.get("skip")),
            "exclude": bool(payload.get("exclude")),
            "rule_label": payload.get("rule_label"),
        }
    return items


@app.get("/api/context_keys")
def api_context_keys():
    """The structured context vocabulary shown in the review form."""
    return {
        "context_keys": list(labeling.CONTEXT_KEYS),
        "feature_map": labeling.CONTEXT_FEATURE_MAP,
        "states": list(labeling.STATE_LABELS),
    }


@app.get("/api/frames")
def api_frames(status: str = "all", offset: int = 0, limit: int = 100):
    """Paginated frame list for the sidebar; ``status`` filters by review state."""
    items = []
    for stem, payload in labeling.iter_scaffolds(LABELING_DIR, load=True):
        if status == "unlabeled" and (payload.get("label") or payload.get("skip") or payload.get("exclude")):
            continue
        if status == "labeled" and not payload.get("label"):
            continue
        if status == "skipped" and not payload.get("skip"):
            continue
        if status == "excluded" and not payload.get("exclude"):
            continue
        if status in ("unlabeled", "labeled", "skipped", "excluded", "all"):
            items.append(stem)
    return {"total": len(items), "offset": offset, "limit": limit, "stems": items[offset : offset + limit]}


@app.get("/api/frame/{stem}")
def api_frame(stem: str):
    """One frame for the review page: scaffold + image path + index within corpus."""
    payload = _scaffold_for(stem)
    image = _image_path_for(stem)
    return {
        "stem": stem,
        "image": f"/api/image/{stem}",
        "image_rel": str(image.relative_to(PROJECT_ROOT)),
        "features": payload,
        "image_width": 2560,
        "image_height": 1440,
    }


@app.get("/api/image/{stem}")
def api_image(stem: str):
    return FileResponse(_image_path_for(stem), media_type="image/png")


@app.post("/api/frame/{stem}")
async def api_save(stem: str, body: dict):
    """Persist an answer: label / skip / exclude / context / silver_override.

    ``context`` holds structured observations (enemy_visible, player_ragdolled,
    ...) that correct misread features at training time; ``silver_override``
    fixes raw Silver values directly. ``exclude`` removes the frame from the
    dataset entirely.
    """
    payload = _scaffold_for(stem)
    if "label" in body:
        label = labeling.valid_label(body["label"])
        if body["label"] is None or str(body["label"]).lower() in ("null", ""):
            label = None
        if label is None and body["label"] is not None:
            raise HTTPException(status_code=400, detail=f"bad label {body['label']!r}")
        payload["label"] = label
        if label is not None:
            payload["skip"] = False  # labeling overrides an earlier skip
    if "skip" in body:
        payload["skip"] = bool(body["skip"])
    if "exclude" in body:
        payload["exclude"] = bool(body["exclude"])
    if "notes" in body:
        payload["notes"] = str(body["notes"])
    if "context" in body and isinstance(body["context"], dict):
        for key, value in body["context"].items():
            if key in labeling.CONTEXT_KEYS:
                payload["context"][key] = None if value is None else bool(value)
    if "silver_override" in body and isinstance(body["silver_override"], dict):
        payload["silver_override"].update(body["silver_override"])
    # Refresh the rule bootstrap on the *corrected* features so the reviewer
    # sees instantly whether an override/context fix changed the state.
    payload["rule_label"] = rule_based_label(apply_corrections(payload, payload))
    saved = labeling.save_scaffold(LABELING_DIR, stem, payload)
    return {"ok": True, "stem": stem, "path": str(saved), "payload": payload}


@app.post("/api/export")
def api_export():
    result = labeling.export_labels_csv(LABELING_DIR, LABELS_CSV)
    return {**result, "summary": labeling.summary(LABELING_DIR)}