# Data Collection Plan

**Game**: Jujutsu Shenanigans (Roblox), 1920×1080 playback
**Goal**: enough real footage for the Gold state-reader (`winning|losing|stalemate`)
to stop leaning on synthetic data. Target **1000–2000 unique, labeled frames**.

This plan answers the three operational questions up front: (1) where video data
goes and how frames get labeled, (2) how to view the MLflow runs, (3) how much
footage you need and how long each video should be.

---

## 1. Where the Data Goes & How It Flows

There is one directory per pipeline layer, so a frame's path tells you its stage:

```
data/videos/<name>.mp4          # raw recordings you add
data/bronze/<stem>.png         # raw screenshots OR frames extracted from a VOD
data/silver/<stem>_bronze.png  # Bronze-preprocessed frame (+ <stem>_bronze.json metadata)
data/silver/<stem>_silver.json # Silver features (player_health, enemies, attacking, ...)
data/gold/silver/*.json        # the same Silver features, pulled into the training set
data/gold/labels.csv           # stem,label  <- source of truth for Gold labels
```

All paths are project-relative and defined in `src/config.py`
(`BRONZE_DIR`, `SILVER_DIR`, `GOLD_DIR`). `.gitignore` keeps captured frames,
models, outputs, and MLflow trees out of git (use `data/**/*.gitkeep`).

### A. Adding video data

1. Drop a **.mp4/.mov/.avi/.mkv/.webm/.flv** recording into `data/videos/`.
2. Extract frames with the Bronze CLI:

```bash
python -m src.pipeline.bronze --input data/videos/fight01.mp4 --output data/silver \
    --frame-interval 60
```

`--frame-interval` counts **source frames**, so use your recording's fps to get
~1 extracted frame per second (60fps recording → `60`, 30fps → `30`). Default
`1` means *every* frame, which at 60fps produces 60× the frames you want.
Frames are written as `fight01_frame_%06d_bronze.png` + a metadata sidecar.

- Videos **over 15 minutes are rejected** (`MAX_VOD_DURATION_SEC`,
  fails gracefully before writing any pixels). Keep recordings short.
- Single screenshots are the zero-friction path: drop PNGs in `data/bronze/`.

### SILVER in one line

```bash
python -c "from src.pipeline.silver import process_frames; process_frames('data/silver/', 'data/gold/silver')"
```

`process_frames` reads every `*_bronze.png` and writes one
`<stem>_silver.json` per frame (health ratio, round index, attacking/defending,
damage flash, timestamps). It will also emit a `rounds.json`. Bronze's VOD
naming and Silver's consumer naming match — both sides use the `_bronze.png`
contract.

### Screenshots only

For a single screenshot: `src.pipeline.silver.process_image(<png>, <out_dir>)`
writes `<stem>_silver.json` directly.

### GOLD trains on real data

```bash
python -m src.pipeline.gold --data-root data/gold --experiment gold_classifier
```

`data/gold/` needs `silver/*.json` (auto-generated above) and `labels.csv`
(ground truth). `--data-root data/gold` is what `train_and_compare()` uses.

---

## 2. Labeling Extracted Frames (Bronze & Silver)

**Bronze needs no labeling.** It is just preprocessed frames — you only *filter*
them (drop menus, loading screens, health-0/dead screens, respawn art screens.
Those are out-of-distribution noise that the label rules cannot judge).

**Silver features are auto-extracted, not hand-labeled.** Each `_silver.json`
already contains the health ratio, enemy count, attacking/defending state, and
OCR confidence. What you hand-label is the *state* of the frame:
`winning | losing | stalemate`.

### Label bootstrap (free labels)

`rule_based_label()` (in `src/pipeline/gold.py`) assigns a provisional state to
every frame from its Silver features — reliable on easy cases (your health full,
theirs slivered → `winning`). It is the starting point for every sample.

### The human-in-the-loop loop

1. **Filter OOD**: drop frames with `player_health == 0.0`, no health bar
   detected, or a rule label that flips frame to frame.
2. **Review a random batch** (~50–100): confirm or correct each rule label and
   append rows to `data/gold/labels.csv` (`stem,label`).
3. **Train the zoo** (`train_and_compare`) on reviewed rows.
4. **Sort the rest by uncertainty** (lowest max-prob, i.e. closest to
   `1/3,1/3,1/3`) and review the most uncertain next. Each iteration shrinks the
   list. This concentrates the hour of hand-labeling on frames the rules/model
   disagree about, instead of re-confirming obvious cases.

`labels.csv` is the single source of truth — `load_labeled_dataset` prefers it
over any embedded `label` key. Expect **~10–20 s/frame** of review.

> **Not yet shipped**: `notebooks/04_label_review.ipynb` (batch reviewer that
> shows frames + rule/model label + type confirm/correct → appends `labels.csv`).
> Until it exists, label directly in a CSV editor or a notebook cell.

### Silver CNN labels (later, optional)

Training the Silver CNN on *real images* needs per-frame annotations (player
health bar region, enemy boxes). Hand-graph ~100–200 frames → train a first
CNN → auto-annotate the rest → verify only uncertain frames (same active loop,
one layer down). Not required for the Gold state-reader.

---

## 3. Viewing the MLflow Runs

### Where runs are stored

Every train/VOD-report run goes to the tracking store controlled by
`MLFLOW_TRACKING_URI` **if set**, otherwise falls back to a SQLite DB under the
temp dir (`sqlite:///tmp/mlruns/mlflow.db` — note `/tmp` is WSL's, wiped on
restart).

**Recommended: point MLflow at a project-local DB once** so runs are durable and
the UI is a single command:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db   # add to your shell rc
```

The `mlruns/` tree (DB + artifacts) is gitignored.

### Start the UI

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# open http://127.0.0.1:5000
```

Use the same URI you exported; if you never exported, use
`sqlite:////tmp/mlruns/mlflow.db`. Runs land in the same store regardless of how
you invoke (CLI, notebook, web app) — `ensure_tracking_uri()` in
`src/pipeline/report.py` guarantees it.

### What you will see

| Look here       | Experiment                | Runs logged by | Metrics | Artifacts |
|---------|---------------------------|----------------|---------|-----------|
| Model quality | `gold_classifier` / `gold_demo` | `train_gold_model`, `train_and_compare` | `accuracy`, `macro_f1`, `weighted_f1`, train counts per label | `confusion_matrix.png`, `classification_report.txt`, logged model |
| Per-VOD reports | `gold_vod_report` | `log_vod_report` (via `process_vod_report`/CLI `--vod-report`) | event counts (`hit_landed`, `punishes`, …), `mean_ocr_confidence`, `score` | `timeline.json`, `stat_card.png` |
| Silver CNN (later) | `silver_cnn` | `train_silver_model` | loss/val loss | checkpoint + params |
| Params | — | every run | `vod_file` (filename), event thresholds, model meta | |

**Model Registry tab**: the trained `state_reader` is registered & versioned
(re-`Staging` → `Production`) by `src/pipeline/model_registry.py`; version the
last run, promote it, and it becomes the one `load_state_reader()` returns.

**Open notebooks**: `run_report_notebook.ipynb` is the interactive scratchpad;
`notebooks/03_gold_modeling.ipynb` exercises the whole Gold layer against MLflow.

---

## 4. How Much Footage, and How Long Each Video

### The ceiling: 15 minutes per VOD

`MAX_VOD_DURATION_SEC = 900`. The pipeline rejects anything longer *before*
framing pixels, so keep every recording under that.

### The arithmetic

- Bronze extracts **~1 frame/sec** (interval = your recording FPS).
- A 10-min video → **~600 raw frames** → after dedup (~5–10× reduction;
  idle moments repeat) → **60–150 unique frames**.
- You need **1000–2000 unique labeled frames**. Therefore:

> **10–20 videos × ~5–10 min each ≈ 2–4 hours of footage** covers the first
> full dataset with a margin.

Per-video guidance:

- **3–10 min, one match set per clip.** Roblox duel rounds run ~2–3 min; keep
  the clip boundary on whole matches so round framing is clean. Don't leave 1-hr
  session captures — dedup overhead explodes and variety per frame drops.
- **Variety is per-file, not per-minute**: flip duels (1v1/2v2), enemy difficulty,
  and maps between clips. One long file is 1000s of similar frames; ten short
  files give diversity for free.
- **Record at 30–60fps.** Extraction samples to ~1 Hz anyway, but damage-flash
  frames (the rarest and most valuable) are only citable if the source recorded
  them.

### Start small

First iteration: **3 clips ≈ 20–30 min total** → run Bronze → Silver → tag the
survivors → `train_and_compare`. Inspect the class balance + uncertainty in
MLflow before scaling to the full 2–4 hours. Keep the first capture phase to ~30
minutes of clips until you know (a) how many frames survive dedup, (b) how
accurate the rule labels are, and (c) what the trainer's F1 looks like. Once
that loop runs in minutes, scale up.

---

## 5. Target Class Distribution

| State     | Target share | Get it by                                                        |
|-----------|--------------|------------------------------------------------------------------|
| winning   | ~40%         | Duels you win; enemy health slivered                            |
| losing    | ~30%         | Duels you lose; fight deliberately stronger foes; **spectate others** |
| stalemate | ~30%         | Opening exchanges, neutral positions, both near-full health      |

Vary also: **match type** (1v1 / 2v2 / messy public server), **health ranges**
(no slivers only — the *ratio* is the signal), **states** (attacking combos,
blocking, hit flashes), **maps & outfits** (they change what "player" looks
like).

---

## 6. Time Budget

| Stage | Time | Output |
|-------|------|--------|
| Play & record (10–15 clips × 5–10 min) | 2–4 hours | 800–2,000 raw frames after extraction+dedup |
| Bronze → Silver over all clips | ~20 min | `*_silver.json` per frame |
| Rule bootstrap + OOD filter | ~10 min | Provisional labels for all frames |
| Human review (uncertainty order) | 2–4 hours | 1,000–2,000 rows in `labels.csv` |
| `train_and_compare` on real data | ~15 min | leaderboard in MLflow `gold_classifier` |

**~6–8 hours calendar time**, most of it play, then review.

---

## 7. Roadmap

| Phase | Deliverable | Done when |
|-------|-------------|-----------|
| 1 | Fresh-volume capture (this doc) | `data/videos/` has 3–30 min of verified-playable clip |
| 2 | Bronze→Silver→gold pass over clips | Frames flow end-to-end, no naming surprises |
| 3 | `notebooks/04_label_review.ipynb` | Confirm/correct labels → `labels.csv` rows append |
| 4 | Label 200–300 (uncertainty order) batches | `train_and_compare` macro-F1 plateaus |
| 5 | Production retrain on `data/gold` | Best model promoted in MLflow registry, `predict()` uses it |
| 6 | (Later) 100–200 hand-annotated Silver frames | First real-data `silver_cnn` run |

---

## 8. Pitfalls

- **Frame-interval trap**: `1` extracts every frame (60fps in = 60 out). Use
  `--frame-interval <recording_fps>` unless you really want every frame.
- **15-minute veto**: respect `MAX_VOD_DURATION_SEC`; cut longer recordings
  rather than fighting the pipeline.
- **Naming contract**: Bronze writes `_bronze.png`, Silver consumes `_bronze.png`
  and writes `_silver.json`. Hand-dropped files that don't follow the contract
  are silently skipped by the globs.
- **Class imbalance**: casual play over-weights stalemate/winning — spectate
  fights you lose, and check `labels.csv` balance each batch.
- **Duplicates**: near-identical idle frames always; dedup before labeling, or
  the model silently overweights idle states.
- **Damage flashes are rare**: capture at 30–60fps even though extraction is 1Hz.
- **OOD frames**: menus / respawn / dead screens be filtered, else the rules
  label nonsense.
- **MLflow location**: the default temp store is wiped on WSL restart — export
  `MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db` before running.
- **Ownership**: capture your own gameplay; don't scrape or redistribute
  others' footage.