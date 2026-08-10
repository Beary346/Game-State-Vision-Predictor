# Game State Vision Predictor

Detect and predict **game state** from fighting-game footage. Upload a match VOD and receive a quantified match report — the match report your eye can't compute.

The pipeline I set up reads raw **Jujutsu Shenanigans (Roblox)** recordings (1920×1080 desktop resolution), extracts HUD-grounded features per frame (health, aggression, defense), and classifies every moment into plain-English initiative states. A single screenshot works too — the zero-friction demo path.

## See it in action

**Label review** — the human-in-the-loop tool that turns raw frames into hand-verified training data:

<img width="1007" height="1024" alt="Labeling UI" src="https://github.com/user-attachments/assets/b60a0933-d386-45b4-a0c1-86496c4b2d68" />

**Gold run #1 — the baseline leaderboard.** The first full model-zoo race on hand-labeled data: six models (logistic regression, random forest, gradient boosting, XGBoost, SVC, PyTorch MLP) trained on **280 hand-labeled frames** with **29 engineered features**. Gradient boosting tops the leaderboard at **0.75 macro-F1**.

<img width="842" height="630" alt="MLflow leaderboard — Gold run #1" src="https://github.com/user-attachments/assets/21e77df3-5068-49b8-ac15-3b70e756132f" />

I believe this score is a floor, not the ceiling: 280 frames is only a modest slice of the 4,008-frame corpus I captured, so the models are still starved for examples of the rarer states (`losing`, `won`, `lost`). Every run after this one benefits from more labels behind it — at 500–1000 samples I expect the leaderboard to tighten and the top F1 to climb, since the labels themselves are human-confirmed.

This run serves as the **baseline**: future runs race the same zoo with the same split and seed, and earn their place as the new Production `state_reader` only by beating it.

<!-- FUTURE RUN: Gold run #2 (500–1000 labeled samples). Paste the new leaderboard screenshot here and compare against the baseline above — the story to tell is the delta: gradient boosting (or whoever takes the top slot) climbing past 0.75 macro-F1 as the dataset grows from 280 → 500+ frames. -->

---

## How it works

A three-stage distillation — each stage turns the data into a higher-level representation, with independent failure modes and feedback loops:

```
Pixels  ──►  Clean Pixels  ──►  Game Features  ──►  State Label
(Bronze)     (Silver)          (Gold)
```

| Layer | What it does | Output |
|-------|--------------|--------|
| **Bronze** | Frame extraction + OpenCV preprocessing (denoise, CLAHE contrast, unsharp mask) | `_bronze.png` per frame |
| **Silver** | OCR-grounded HUD reads: player/enemy health, positions, attacking/defending/damage state, OCR confidence | `_silver.json` per frame |
| **Gold** | MLflow races a model zoo (scikit-learn / XGBoost / PyTorch), picks the winner, and serves the **`state_reader`** production model | Timeline + stat card + MLflow run per VOD |

## What you get per match

- **Event timeline** — health-drop hits, punishes (no return hit), round losses, domain deployments
- **Hit / whiff stats and damage-by-round breakdown**
- **A downloadable stat card** with a headline **score** (timeline plot + headline stats), built with PIL and readable at screen-share zoom
- Every processed VOD becomes an **MLflow run** with params, metrics, and artifacts (timeline JSON + stat card)

---

## Getting started

Requires **Python 3.11+**. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[ml,viz,workflow,dev]"

# Point MLflow at a project-local store once per shell (add to your rc):
export MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db
```

## What the states mean

The Gold layer labels each frame by the player's **local initiative** — not health advantage. A 1-HP player landing hits is `winning`; a full-HP player trapped in a combo is `losing`.

| Label | Meaning |
|-------|---------|
| `winning` | The player is currently striking the enemy |
| `losing` | The player is currently being hit (takes priority on overlap) |
| `stalemate` | Neutral / defending — nothing is happening |
| `searching` | *Looking for the enemy between exchanges* |
| `won` | *The round just ended in the player's favor* |
| `lost` | *The round just ended against the player* |

The italicized states are extended meanings used by the review tool; the core trained states are `winning | losing | stalemate`.

## Labeling — human-in-the-loop

Silver features are auto-extracted; you hand-label each frame's *state* with the web review tool. It shows the Bronze frame, streams its Silver features plus a rule-bootstrap provisional label, and writes your answer straight back into the per-frame scaffold.

```bash
python -m src.pipeline.labeling --init     # one empty scaffold per Silver JSON
uvicorn app.labeler:app --port 8765        # open http://127.0.0.1:8765
```

**Keyboard shortcuts:** `1/2/3/4/5/6` → winning / losing / stalemate / searching / won / lost, `s` skip (out-of-distribution), `x` exclude from training, `u` undo, `n`/`p` next/previous, `e` export. Filling a label auto-advances to the next unlabeled frame.

When Silver misreads a frame, fix it directly in the UI: adjust the `silver_override` values (e.g. `player_health`) — the rule label refreshes instantly and training consumes the correction, while `labels.csv` remains the single source of truth.

```bash
python -m src.pipeline.labeling --export   # writes data/gold/labels.csv
```

> **Keep the dataset clean:** drop out-of-distribution noise in review — menus, loading screens, death screens, zero-health frames. The label rules can't judge those.

### The whole flow, from raw match to trained model

```
data/videos/*.mp4 ──①──► data/silver/*_bronze.png ──②──► data/gold/silver/*_silver.json
                        (bronze)                     (silver)

──③──► data/gold/labeling/*_labeling.json ──④──► review in browser ──⑤──► labels.csv
      (scaffolds)                              (uvicorn app.labeler:app)

──⑥──► MLflow gold_classifier leaderboard ──► Model Registry state_reader (Production)
      (python -m src.pipeline.gold)
```

①、② rerun only when you add new VODs; ③→⑥ repeat as you label more.

---

## Train & run the state-reader

```bash
# ① Race every model in the zoo, log runs to MLflow, print the leaderboard
python -m src.pipeline.gold --data-root data/gold

# …or auto-promote the winner to the versioned Production state-reader
python -m src.pipeline.gold --data-root data/gold --register-best

# ② Run the production model on a VOD it has never seen
python -m src.pipeline.gold --infer-vod data/videos/your_new_match.mp4 \
    --experiment gold_vod_report
# → outputs/gold_<name>/timeline.json + stat_card.png, logged as an MLflow run

# ③ Inspect everything in the UI
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # http://127.0.0.1:5000
```

Useful flags: `--infer-interval N` samples every N frames (default: the VOD's fps = ~1 frame per second); `--infer-model models:/state_reader/3` overrides the model URI. No registered model yet? `--infer-vod` falls back to the rule-based bootstrap labels until the first `--register-best`.

Programmatic equivalent:

```python
from src.pipeline.gold import infer_new_vod, register_best_model, train_and_compare

leaderboard = train_and_compare("data/gold")
register_best_model(leaderboard)
report = infer_new_vod("data/videos/fresh_match.mp4")
```

## CLI cheat sheet

| Goal | Command |
|------|---------|
| Extract frames from a VOD | `python -m src.pipeline.bronze --input data/videos/f1.mp4 --output data/silver --frame-interval 60` |
| Silver features over all frames | `python -c "from src.pipeline.silver import process_frames; process_frames('data/silver/', 'data/gold/silver')"` |
| Single screenshot → features | `python -c "from src.pipeline.silver import process_image; process_image('shot.png', 'out/')"` |
| Label scaffolds | `python -m src.pipeline.labeling --init` — also `--refresh-rules`, `--summary`, `--export` |
| Label review UI | `uvicorn app.labeler:app --port 8765` |
| Train the zoo / promote best | `python -m src.pipeline.gold --data-root data/gold [--register-best]` |
| Match report on a new VOD | `python -m src.pipeline.gold --infer-vod <file>.mp4 --experiment gold_vod_report` |
| MLflow UI | `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` |
| Silver CNN (later milestone) | `python -m src.pipeline.silver_train --synthetic 200 --epochs 50` |

---

## Running with Docker

Everything above (label review, training, MLflow UI) also runs containerized:

```bash
docker compose up -d                      # MLflow UI :5000 + label review :8765
# label frames in the browser, then:
docker compose --profile training run --rm training   # train + register state_reader
# open http://127.0.0.1:5000 -> gold_classifier leaderboard + Model Registry
```

- Labels live in `./data` (bind mount) — the training job reads exactly what you labeled.
- MLflow runs and artifacts persist in the `mlflow-mlruns` / `mlflow-artifacts` volumes (same store the app and training job share via the `mlflow` service).

## Running the tests

```bash
pytest          # bronze, silver, gold, events, report, registry, labeling
ruff check .    # lint
```

---

## Project layout

```
├── src/
│   ├── ingestor_bronze.py         # frame extraction (VOD → PNGs)
│   ├── classifier_silver.py       # OCR heuristics → state tuples
│   ├── pipeline/
│   │   ├── bronze.py              # CLI shim
│   │   ├── silver.py              # per-frame Silver features + rounds.json
│   │   ├── labeling.py            # scaffolds + labels.csv export
│   │   ├── gold.py                # model zoo, leaderboard, train/infer CLI
│   │   ├── events_gold.py         # state deltas → hit / punish / round-loss events
│   │   ├── report.py              # timeline, score, MLflow logging
│   │   └── renderer_reportcard.py # PIL stat card
│   ├── models/silver_cnn.py       # multi-head CNN (later milestone)
│   └── utils/image_utils.py
├── app/
│   ├── labeler.py                 # FastAPI label-review server
│   └── static/index.html          # the review UI
├── tests/                         # tests written before implementation
├── docs/                          # design + data collection plan
└── notebooks/                     # interactive visualization (01–03)
```

## Data conventions

- Record at **1920×1080** and **keep clips under 15 minutes** — the pipeline rejects longer VODs before writing a single pixel.
- `--frame-interval` counts *source* frames (`60` for 60fps → ~1 extracted frame per second).
- Raw VODs and extracted frames are **gitignored** — footage lives in `data/videos/`, never in git.

## Documentation

- `docs/design.md` — the what, the why, and the trade-offs behind every major decision
- `docs/data_collection_plan.md` — the human loop: capturing, labeling, training, and the target dataset

## Tools & acknowledgments

Built with **OpenCode** as an AI coding assistant. The architecture and guidance skills behind this project were learned from the **Databricks** documentation, the **NVIDIA** documentation, and personal experience in machine-learning competitions.

---

## Future screenshots

<!-- FUTURE IMAGE: Downloadable stat card (outputs/gold_<name>/stat_card.png).
<img width="1254" height="1390" alt="Sample stat card" src="https://github.com/user-attachments/assets/00000000-0000-0000-0000-000000000000" /> -->

<!-- FUTURE IMAGE: Timeline plot from a VOD report (gold_vod_report artifacts).
<img width="1254" height="1390" alt="Sample match timeline" src="https://github.com/user-attachments/assets/00000000-0000-0000-0000-000000000000" /> -->
