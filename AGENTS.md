# Jujutsu Shenanigans Match Analyzer

## Objective
Detect and predict "game state" combinations from fighting game footage. A free tool where players upload a match VOD and receive a quantified match report — the match report your eye can't compute.

## Inputs
Raw match VODs of **Jujutsu Shenanigans (Roblox)**, recorded at playback resolution H=1920, W=1080 (Standard desktop resolution). A single-frame "state check" mode (single screenshot) stays as the zero-friction demo entry.

## Outputs
Machine output categorized as `winning`, `losing`, or `stalemate`, based on the per-frame game state:
- **Health ratio** (player vs enemy)
- **Aggression** (is the player attacking?)
- **Defense** (how is the player defending?)

Delivered as a quantified match report:
- **Event timeline** (health-drop hits, whiffs, punishes, round losses, domain deployments)
- **Hit/whiff stats and damage-by-round breakdown**
- A **downloadable, shareable stat card that gives you a score** (timeline plot + headline stats + scoring system for dopamine hit)

## Pipeline

### Bronze (Raw VOD / Screenshot)
Frame extraction and OpenCV preprocessing. Handle single-frame screenshots and multi-minute VODs uniformly.

### Silver (HUD Grounding per Frame)
OCR reads grounded HUD regions on each frame — health bar, round/clock, ability/domain indicators. Player identification: player is always relatively center screen, has no name above their head, and no health bar on their head (this distinguishes them from enemies). Clean per-frame state tuples that aggregate into per-round state.

### Event Layer (State-Delta Aggregation)
Changes between consecutive state tuples become events:
- **Health drop** = hit landed
- **No return hit** = punish
- **Death** = round loss
- **Domain alert** = domain deployed

### Gold (Trained Model + Reporting)
MLflow tests a wide variety of models to find the best fit. Trains a model using plain-English category labels derived from Silver data. Every processed VOD is an MLflow run with parameters, metrics, and artifacts (timeline JSON + stat card).

## Constraints
- **Tensor shape assertions**: enforce `assert img.shape == (H, W, 3)` to prevent silent broadcasting bugs. H=1920, W=1080 (Standard desktop resolution) to ensure all data is captured.
- **Reasonably fails on expected bad inputs**: VODs over ~15 minutes fail gracefully, OCR-confidence warnings are surfaced, and unusual resolutions are handled without crashing.
- **One Game structure**: train on a single game initially to avoid conflicting noise. Game: **Jujutsu Shenanigans (Roblox)** — chosen for its high attack variation and familiarity.

## Core Features
- OpenCV for data extraction and frame preprocessing
- HUD-region OCR grounding per frame (health bar, round/clock, ability/domain indicators) with known read accuracy on held-out frames
- Effective noise removal (distinguish important data from noise)
- Event timeline generation from per-frame state deltas
- Final state classification (winning/losing/stalemate) from Gold data
- Per-VOD MLflow run: params (VOD file, OCR thresholds), metrics (event counts, mean OCR confidence), artifacts (timeline JSON + stat card)
- State-reader registered as a versioned model; stat card generated with PIL (readable at screen-share zoom)
- A web UI built with React for drag-and-drop VOD upload with job progress, then serving for report and stat-card downloading

## Stack

| Category       | Tool           | Purpose                      |
|----------------|----------------|------------------------------|
| Assistance     | OpenCode       | Coding assistance            |
| Modeling       | Scikit-learn   | Models / Train-Test Data     |
| Math           | NumPy          | Math operations              |
| Data           | Pandas        | Data manipulation            |
| Workflow       | MLflow         | ML workflow organization     |
| Image          | OpenCV         | Image processing             |
| Visualization  | JupyterLab     | Data visualization            |
| Visualization  | Matplotlib     | Data visualization            |
| Visualization  | Seaborn        | Statistical visualization    |
| Modeling       | PyTorch        | Deep learning models         |
| Modeling       | XGBoost       | Gradient boosting models     |
| Foundation     | PyTorch       | Deep learning models        |
| Testing        | pytest        | In-Depth tests               |
| Data Extraction | Analytics | Temporal/Ascenders | Video frame extraction |

## Project structure (brick tracker)

```
game_state_vision_predictor/
├── .github/
│   └── workflows/   # auto-test runs on push, lint + smoke test on tiny VOD
├── src/
│   ├── ingestor_bronze.py        # frame extraction / bronze team
│   ├── classifier_silver.py      # OCR heuristics → state tuples
│   ├── events_gold.py            # state-delta events
│   ├── report.py                 # timeline + stat card
│   ├── renderer_reportcard.py    # PIL stat card
│   ├── euclidean_ml.py           # feature engineering + SKLearn models
│   └── model_registry.py         # MLflow model versioning
├── app/
│   ├── api.py                    # FastAPI upload → async job status
│   └── static/
│       ├── index.html
│       └── ...                    # drag-drop + timeline rendering
├── tests/                        # tests written BEFORE implementation
└── run_report_notebook.ipynb      # interactive
```

## Rules and Agreements
- Write tests before implementation.
- Load from AGENTS.md, Stay commit to the product promise, Ask questions.
- Three golden rules: Reason, Accuracy first, combination prove, correctness first.
- Code coverage: build it, then add tests later (`XD: 10 events` maybe easily that). Optimize later, correctness first.
- Plan the trials as CLI + notebook + FastAPI (web UI) for interactive / human-in-the-loop use.
- Final output plat requires: **stream time-series now?**
- Add comments on your code, prioritizing developer readability: concise docstrings, and descriptive variable names can replace the need for excessive inline comments.
- Don't just log numbers: log correlation matrices, UMAP/t-SNE embeddings, and confusion matrices. Take advantage of JupyterLab, MLflow, Matplotlib, and a Web UI (or notebook) for interactive, human-in-the-loop use.