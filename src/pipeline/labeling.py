"""Labeling scaffolds: per-frame Gold training scaffolds.

One ``<stem>_labeling.json`` is written for every real Silver frame. It starts
as a verbatim copy of the Silver feature tuple (the shape a reviewer wants to
see: ``player_health``, ``enemies``, ``attacking``, ...) plus the fields the
reviewer fills in:

- ``label``          one of the six states (winning | losing | stalemate |
                     searching | won | lost) or ``null``
- ``skip``           true when the frame is out-of-distribution noise
- ``exclude``        true when the frame must be removed from the dataset
                     (unclear / unwanted footage — the "remove frames while
                     labeling" escape hatch)
- ``context``        structured, ML-relevant observations about the frame
                     (enemy visible? ragdolled? ultimate active?) — the
                     focused context that replaces free-form notes
- ``silver_override`` corrections of misread Silver values (full control over
                     what the model sees in training)

The web tool in ``app/labeler.py`` shows the bronze image for each stem,
exposes the extracted features + the rule-bootstrap label, and writes answers
straight back into these JSON files. ``export_labels_csv`` is the bridge to
Gold: it flattens every labeled, non-skipped, non-excluded scaffold into
``labels.csv``, which ``src/pipeline/gold.py`` treats as the source of truth
(and its ``load_labeled_dataset`` additionally applies ``silver_override`` +
``context`` corrections when building the feature matrix).

CLI::

    python -m src.pipeline.labeling --init                           # create scaffolds
    python -m src.pipeline.labeling --summary                        # progress counts
    python -m src.pipeline.labeling --export                         # build labels.csv

Paths follow data_collection_plan.md::

    data/gold/silver/<stem>_silver.json        # Silver features (pre-existing)
    data/gold/labeling/<stem>_labeling.json   # this module's scaffold
    data/gold/labels.csv                      # stem,label  <- Gold source of truth
"""

import argparse
import csv
import json
from pathlib import Path

from src.pipeline.gold import (
    CONTEXT_FEATURE_MAP,
    CONTEXT_NOTES,
    apply_corrections,
    rule_based_label,
)

# The six plain-English states; order matches gold.STATE_LABELS.
STATE_LABELS: tuple[str, ...] = (
    "winning",
    "losing",
    "stalemate",
    "searching",
    "won",
    "lost",
)

# Context fields the reviewer can confirm/correct. Feature-mapped ones
# (ragdoll, ultimate, domain) feed the model as corrected features; the
# situational ones (enemy_visible, player_moving, round start/end) inform the
# label choice and are kept for the reviewer, not vectorized.
CONTEXT_KEYS: tuple[str, ...] = tuple(CONTEXT_FEATURE_MAP) + CONTEXT_NOTES


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def labeling_stem(silver_stem: str) -> str:
    """Map a Silver filename stem to its labeling stem (``<stem>_silver`` -> ``<stem>``)."""
    return silver_stem.removesuffix("_silver")


def build_scaffold(silver_json: dict, stem: str) -> dict:
    """Build an empty labeling scaffold from a Silver feature tuple.

    ``silver_json`` is copied verbatim so nothing between the feature layer and
    the label is lost; ``rule_label`` is the deterministic rule bootstrap from
    Gold so the reviewer only has to confirm or correct obvious cases. The
    reviewer-facing fields are all empty: ``label``, ``skip``, ``exclude``,
    ``context`` (structured observations), ``silver_override`` (value fixes).
    """
    scaffold = dict(silver_json)
    scaffold["label"] = None
    scaffold["skip"] = False
    scaffold["exclude"] = False
    scaffold["context"] = {}
    scaffold["silver_override"] = {}
    scaffold["rule_label"] = rule_based_label(scaffold)
    scaffold["labeling_stem"] = stem
    return scaffold


def init_labeling_files(silver_dir: str | Path, labeling_dir: str | Path) -> dict:
    """Create an empty labeling scaffold for every Silver JSON.

    Files that already exist are never overwritten, so labels survive re-runs.
    Returns ``{"created", "existing", "total"}``.
    """
    silver_dir = Path(silver_dir)
    labeling_dir = Path(labeling_dir)
    created = existing = 0
    for json_path in sorted(silver_dir.glob("*_silver.json")):
        data = _read_json(json_path)
        if "player_health" not in data:
            continue  # not a Silver features file
        stem = labeling_stem(json_path.stem)
        out = labeling_dir / f"{stem}_labeling.json"
        if out.exists():
            existing += 1
            continue
        _write_json(out, build_scaffold(data, stem))
        created += 1
    return {"created": created, "existing": existing, "total": created + existing}


def _normalize_scaffold(payload: dict) -> dict:
    """Guarantee the scaffold keys exist even on hand-edited files."""
    payload.setdefault("label", None)
    payload.setdefault("skip", False)
    payload.setdefault("exclude", False)
    context = payload.setdefault("context", {})
    for key in CONTEXT_KEYS:
        context.setdefault(key, None)
    payload.setdefault("silver_override", {})
    payload.setdefault("rule_label", None)
    return payload


def load_scaffold(labeling_dir: str | Path, stem: str) -> dict:
    """Load one labeling scaffold (raises FileNotFoundError when missing)."""
    path = Path(labeling_dir) / f"{stem}_labeling.json"
    if not path.exists():
        raise FileNotFoundError(f"No labeling file for stem {stem!r} at {path}")
    return _normalize_scaffold(_read_json(path))


def save_scaffold(labeling_dir: str | Path, stem: str, payload: dict) -> Path:
    """Persist a labeling scaffold back to disk."""
    path = Path(labeling_dir) / f"{stem}_labeling.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)
    return path


def iter_scaffolds(labeling_dir: str | Path, load: bool = True) -> list[tuple[str, dict]]:
    """List ``(stem, payload)`` for every scaffold in the labeling directory.

    With ``load=False`` payloads are empty dicts -- cheaper when only filenames
    matter (total frame count, frame list).
    """
    labeling_dir = Path(labeling_dir)
    result: list[tuple[str, dict]] = []
    for path in sorted(labeling_dir.glob("*_labeling.json")):
        stem = path.stem.removesuffix("_labeling")
        payload = _normalize_scaffold(_read_json(path)) if load else {}
        result.append((stem, payload))
    return result


def refresh_rule_labels(labeling_dir: str | Path) -> dict:
    """Recompute ``rule_label`` on every scaffold in place.

    Scaffolds are copies of Silver features plus reviewer fields, so the rule
    bootstrap can be recomputed straight from the scaffold dict. Human
    ``label`` / ``skip`` / ``exclude`` / ``context`` / ``silver_override``
    answers are left untouched — but the rule itself is evaluated on the
    *corrected* features (silver_override + context applied), so a reviewer
    fixup that changes the state shows up in the bootstrap.
    """
    labeling_dir = Path(labeling_dir)
    refreshed = 0
    for stem, payload in iter_scaffolds(labeling_dir, load=True):
        corrected = apply_corrections(payload, payload)
        new_rule = rule_based_label(corrected)
        if payload.get("rule_label") != new_rule:
            payload["rule_label"] = new_rule
            _write_json(labeling_dir / f"{stem}_labeling.json", payload)
            refreshed += 1
    return {"refreshed": refreshed, "total": len(iter_scaffolds(labeling_dir, load=False))}


def valid_label(value) -> str | None:
    """``value`` normalized and returned when it is one of STATE_LABELS, else ``None``."""
    candidate = str(value).strip().lower()
    return candidate if candidate in STATE_LABELS else None


def summary(labeling_dir: str | Path) -> dict:
    """Progress counts for the labeling corpus."""
    total = labeled = skipped = excluded = 0
    by_label = {label: 0 for label in STATE_LABELS}
    for _, payload in iter_scaffolds(labeling_dir, load=True):
        total += 1
        if payload.get("exclude"):
            excluded += 1
            continue
        if payload.get("skip"):
            skipped += 1
            continue
        label = payload.get("label")
        if label in by_label:
            labeled += 1
            by_label[label] += 1
    return {
        "total": total,
        "labeled": labeled,
        "skipped": skipped,
        "excluded": excluded,
        "unlabeled": total - labeled - skipped - excluded,
        "by_label": by_label,
    }


def export_labels_csv(labeling_dir: str | Path, csv_path: str | Path) -> dict:
    """Flatten labeled, non-skipped, non-excluded scaffolds into labels.csv.

    ``exclude`` is the reviewer's "remove this frame from the dataset" flag:
    it clears the sample from training data entirely (and from the CSV).
    """
    rows = []
    for stem, payload in iter_scaffolds(labeling_dir, load=True):
        if payload.get("exclude") or payload.get("skip"):
            continue
        label = payload.get("label")
        if label not in STATE_LABELS:
            continue
        rows.append({"stem": stem, "label": label})

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stem", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "path": str(csv_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gold labeling scaffold tooling")
    ap.add_argument("--init", action="store_true", help="create empty scaffolds for every Silver JSON")
    ap.add_argument("--refresh-rules", action="store_true", help="recompute rule_label on existing scaffolds")
    ap.add_argument("--summary", action="store_true", help="print labeling progress counts")
    ap.add_argument("--export", action="store_true", help="write labels.csv from labeled scaffolds")
    ap.add_argument("--silver", default="data/gold/silver", help="Silver JSON directory")
    ap.add_argument("--labeling", default="data/gold/labeling", help="labeling scaffold directory")
    ap.add_argument("--out", default="data/gold/labels.csv", help="labels.csv output path")
    args = ap.parse_args()

    if args.init:
        res = init_labeling_files(args.silver, args.labeling)
        print(
            f"Scaffolds: {res['created']} created, {res['existing']} unchanged, {res['total']} total"
        )
    if args.refresh_rules:
        res = refresh_rule_labels(args.labeling)
        print(f"Rule labels: {res['refreshed']} updated, {res['total']} scaffolds present")
    if args.summary:
        res = summary(args.labeling)
        print(
            f"Progress: {res['labeled']}/{res['total']} labeled, {res['skipped']} skipped, "
            f"{res['excluded']} excluded, {res['unlabeled']} remaining"
        )
        print(f"  by label: {res['by_label']}")
    if args.export:
        res = export_labels_csv(args.labeling, args.out)
        print(f"Exported {res['rows']} labeled rows -> {res['path']}")