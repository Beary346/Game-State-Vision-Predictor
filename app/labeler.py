"""FastAPI label-review web tool for Gold Silver frames.

Run from the project root::

    uvicorn app.labeler:app --host 127.0.0.1 --port 8765
    # open http://127.0.0.1:8765

Workflow:
    1. ``python -m src.pipeline.labeling --init``  (creates the empty scaffolds,
       already done for the current corpus)
    2. Start the server and open the URL.
    3. Review frames: see the bronze frame, its Silver features and the
       rule-bootstrap label, then click ``winning / losing / stalemate`` (or
       ``Skip`` for out-of-distribution noise). Every answer is written straight
       back into ``data/gold/labeling/<stem>_labeling.json``.
    4. Hit ``Export labels.csv`` once done (or ``/api/export``) -- this writes
       ``data/gold/labels.csv`` from every labeled, non-skipped scaffold.
    5. Train Gold: ``python -m src.pipeline.gold --data-root data/gold``

Keyboard shortcuts (from the review page):
    ``1/2/3``  label winning/losing/stalemate
    ``s``      toggle skip / out-of-distribution
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
            "rule_label": payload.get("rule_label"),
        }
    return items


@app.get("/api/frames")
def api_frames(status: str = "all", offset: int = 0, limit: int = 100):
    """Paginated frame list for the sidebar; ``status`` filters by review state."""
    items = []
    for stem, payload in labeling.iter_scaffolds(LABELING_DIR, load=True):
        if status == "unlabeled" and (payload.get("label") or payload.get("skip")):
            continue
        if status == "labeled" and not payload.get("label"):
            continue
        if status == "skipped" and not payload.get("skip"):
            continue
        if status in ("unlabeled", "labeled", "skipped", "all"):
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
    """Persist an answer: ``{"label": "winning|...|null", "skip": bool, "notes": str}``."""
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
    if "notes" in body:
        payload["notes"] = str(body["notes"])
    saved = labeling.save_scaffold(LABELING_DIR, stem, payload)
    return {"ok": True, "stem": stem, "path": str(saved), "payload": payload}


@app.post("/api/export")
def api_export():
    result = labeling.export_labels_csv(LABELING_DIR, LABELS_CSV)
    return {**result, "summary": labeling.summary(LABELING_DIR)}