# Data Collection Plan: 1000–2000 Labeled Screenshots

**Game**: Jujutsu Shenanigans (Roblox)
**Goal**: 1000–2000 real, labeled screenshots so the Gold classifier (and later
the Silver CNN) stops relying on synthetic data.

---

## 1. The Contract

Every screenshot must end up as a labeled sample the pipeline already
understands:

```
capture → data/bronze/*.png          (raw screenshots)
        → data/silver/*.json         (Silver features, via src.pipeline.silver)
        → labels.csv                 (stem,label → "winning"|"losing"|"stalemate")
        → data/gold/                 (consumed by src.pipeline.gold)
```

The Gold layer already handles everything after capture: `process_image` writes
the feature JSONs, `load_labeled_dataset` reads `labels.csv`, and
`train_and_compare` trains + logs the leaderboard. **The only missing piece is
the labeled screenshots themselves.**

---

## 2. What "Good Data" Looks Like

Capture diversity is the single biggest quality lever. Target rough class
distribution for Gold:

| State      | Target share | How to get it                                    |
|------------|--------------|--------------------------------------------------|
| winning    | ~40%         | Duels you win; low enemy health moments          |
| losing     | ~30%         | Duels you lose; deliberately fight stronger foes; spectate others' fights |
| stalemate  | ~30%         | Opening exchanges, both at full health, neutral positions |

Also vary the hidden variables the Silver CNN must learn:

- **Match type**: 1v1, 2v2, and public-server chaos (1–3 enemies on screen)
- **Health ranges**: full, half, slivers — the ratio matters, not just extremes
- **States**: attacking (combos, ultimates), defending (blocking), damage
  flashes (screen flash right as you get hit — these frames are rare, so the
  capture rate matters)
- **Visual variety**: different maps (Grass Field, Gym, Sewers, Tombs of the
  Star...), different times of day if the map lighting changes, different
  outfits (they change what "player" looks like to the CNN)

**Filter OUT**: menus, loading screens, respawn/cutscene screens, and dead
screens (player health 0). These are out-of-distribution noise.

---

## 3. Capture Options (Discovery)

| Option | Automation | Throughput | Effort | Verdict |
|--------|-----------|------------|--------|---------|
| **Windows screen-capture script (mss)** | Full auto while you play | 1 fps → ~2000+ unique frames/hour | Low | **Recommended** |
| **Xbox Game Bar / OBS recording → ffmpeg frames** | Full auto, no script during play | Any fps you want | Low | Best passive option; also captures damage-flash frames reliably |
| Roblox built-in F12 screenshot (`C:\Users\<you>\Pictures\Roblox`) | Manual | ~1 per keypress | Trivial | Good for spot-checking, too slow alone |
| Spectate other players' fights | Manual (spectate after death) | Fast, diverse | Zero risk | Great source of **losing** frames |
| Ranked/Casual Duels mode | Manual play | Controlled 1v1/2v2 | Low | Best for clean, balanced data |
| YouTube/Twitch gameplay videos | Extract frames with ffmpeg | Huge | Copyright risk | **Supplement only** — not needed if you play |

**Recommendation**: play Duels + public servers for ~1–2 hours total with the
capture script running at 1 fps, and record with Xbox Game Bar or OBS for the
same sessions as a fallback. Two capture sessions yield more raw frames than
you need; dedup and filtering do the rest.

---

## 4. Recommended Automated Capture Pipeline

### 4.1 Capture (Windows side — Roblox runs on Windows, not WSL)

New script: `scripts/capture_screenshots.py`, run with the **Windows** Python
(`C:\Users\<you>\AppData\Local\Programs\Python\Python311\python.exe`), pip
install `mss pynput`:

- Grabs the full monitor (or just the Roblox window region) at a configurable
  interval (default 1 fps — fast enough to catch damage flashes, slow enough
  to avoid 10,000 near-identical frames)
- Saves PNGs directly into the shared folder
  `C:\Users\<you>\Documents\Coding\Game State Vision Predictor\data\capture\raw\`
  so WSL sees them instantly (no copying)
- Hotkeys while playing: **P** = pause, **S** = save a manual snapshot,
  **Q** = quit
- Skips a frame if it is nearly identical to the previous one (mean absolute
  pixel difference below a threshold) — free dedup at capture time

### 4.2 Extraction (if you recorded instead)

One-liner per recording:

```bash
ffmpeg -i recording.mp4 -vf fps=1 -q:v 2 data/capture/raw/frame_%06d.png
```

Then the same dedup pass (script `scripts/dedupe_frames.py` or a notebook
cell) removes near-duplicates via a perceptual hash / pixel-diff threshold.
Expect a ~5–10x reduction from raw frames to unique states.

### 4.3 Filter & move into the pipeline

Once captured, every frame flows through existing code with zero new tooling:

```bash
# move raw captures into the bronze intake
cp data/capture/raw/*.png data/bronze/
python -m src.pipeline.gold --data-root data/gold --synthetic 0  # (sanity check only)
```

Run Bronze + Silver on all frames (Silver defaults are fine for now — the
`player_health`/`num_enemies`/`attacking` fields drive the next step), then
drop frames whose features are obviously OOD (player_health == 0.0, no health
bar detected, or rule output flips every frame). A small notebook cell does
this; keep the survivors in `data/gold/silver/`.

---

## 5. Labeling Strategy — The Efficiency Plan

Hand-labeling 2000 screenshots from scratch is slow. The plan makes humans
label only the ~300–500 hardest frames:

### Step 1 — Rule bootstrap (free labels)
`rule_based_label()` already assigns `winning/losing/stalemate` to every frame
from the Silver features. It is right on easy cases (your health 100, theirs
10 → "winning"). This is the starting point for every sample:
`gold/silver/*.json` gets an embedded `label` key automatically.

### Step 2 — Human review with active learning (label the informative ones)
1. Review a small random batch (50–100 frames) in the review UI — confirm or
   correct each rule-based label.
2. Train the Gold zoo on reviewed data (`train_and_compare`).
3. Predict the remaining unlabeled frames with the best model and **sort by
   uncertainty** (probability closest to 0.33/0.33/0.33, or lowest max-prob).
4. Review the most uncertain frames next — each correction retrains the
   model, so the uncertain list keeps shrinking.

This concentrates human effort where the rules and model disagree, instead of
re-verifying obvious cases.

### Step 3 — Review tooling (next deliverable)
A `notebooks/04_label_review.ipynb` batch-reviewer: shows N screenshots with
the rule label + model probabilities, you type confirm/correct, it appends
`labels.csv`. Approx 10–20 s per screenshot → 300 corrections ≈ 1–2 hours.

### Step 4 — Silver annotations (second pass, later)
The Gold labels get you a working classifier. To train the Silver CNN on real
images you need per-frame annotations (player health, enemy bboxes) — too
slow to hand-draw. Plan: hand-annotate ~100–200 frames (a rectangle per
health bar, a flag per state) → train a first real SilverCNN → use it to
auto-annotate the remaining 2000+ frames → human-verify only frames where the
CNN is uncertain. Same active-learning loop, one layer down.

---

## 6. Time & Throughput Budget

| Stage | Time | Output |
|-------|------|--------|
| Capture (2 sessions of play, 1 fps + recording) | 2–3 hours | 5,000–10,000 raw frames |
| Dedup + filter (automated) | ~15 min | 1,500–2,500 unique usable frames |
| Rule bootstrap + auto-feature extraction | ~15 min | All frames labeled (provisionally) |
| Human review, active-learning order | 2–4 hours | 1,000–2,000 confirmed labels |
| First real Gold model on real data | 15 min | Leaderboard in MLflow |

**Total: ~6–9 hours of calendar time**, of which ~4 are active play and ~2–4
are focused labeling. Well inside a week of evening sessions.

---

## 7. Roadmap

| Phase | Deliverable | Done when |
|-------|-------------|-----------|
| 1 | `scripts/capture_screenshots.py` (Windows Python) | Captures 1 fps + hotkeys, writes PNGs to `data/capture/raw/` |
| 2 | 30-min test capture + dedup script | ~1,000 unique frames survive dedup |
| 3 | `notebooks/04_label_review.ipynb` review UI | Confirm/correct labels → `labels.csv` rows append |
| 4 | Label in batches of 200–300 (uncertainty order) | `train_and_compare` macro-F1 stops improving (plateau) |
| 5 | Production retrain on `data/gold` | Best model logged to MLflow; used by `predict()` |
| 6 | (Later) 100–200 hand-annotated Silver frames | First real-data SilverCNN training run |

---

## 8. Pitfalls

- **WSL vs Windows split**: Roblox only runs on Windows. Capture scripts must
  run under Windows Python; WSL does all processing on the shared drive. Do
  not try to capture from WSL.
- **Class imbalance**: casual play is mostly stalemate/winning — losing frames
  are the rare, most valuable ones. Spectate fights you lose, fight ranked
  opponents, and check the class balance plot after every batch.
- **Duplicate frames**: at 1 fps most frames still repeat (idle moments).
  Always dedup; otherwise the model silently overweights idle states.
- **Damage-flash frames**: they last a few frames — 1 fps might miss most of
  them. The video-recording fallback (Xbox Game Bar) exists precisely for this.
- **Out-of-distribution frames**: menus, respawn screens, and health-0 frames
  will be labeled nonsense by the rules. Filter them before review.
- **Resolution**: the Bronze layer resizes to 2560×1440, but capture at that
  native resolution if possible (set monitor to 2560×1440 while playing) so
  health bars keep full detail.
- **Ownership**: only capture your own gameplay for training; do not scrape
  and redistribute others' content.
