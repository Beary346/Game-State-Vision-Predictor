# System Design: Game State Vision Predictor

## Why This Document Exists

This document is for when you want to **internalize** how the system works — to
hold the architecture in your head without running code. It describes the what,
the why, and the trade-offs behind every major decision. Notebooks show you the
code in action; this document shows you the thinking behind it.

---

## 1. The Problem

**Input**: A raw game screenshot (2560 × 1440 pixels).
**Output**: One of three labels — `winning`, `losing`, or `stalemate`.

The challenge is that a screenshot is just a grid of RGB numbers. There is no
direct way to ask "how much health does the player have?" or "is the player
attacking?" — those are **latent game-state variables** that must be inferred
from pixel patterns.

---

## 2. The Pipeline: Bronze → Silver → Gold

The pipeline is a **three-stage distillation**. Each stage transforms the data
into a higher-level representation:

```
Pixels  ──►  Clean Pixels  ──►  Game Features  ──►  State Label
(Bronze)      (Silver)          (Gold)
```

### Why three stages? Why not an end-to-end model?

Because each stage has a different **failure mode** and **feedback loop**:

| Stage | What fails | How you fix it |
|-------|-----------|----------------|
| Bronze | Salt-and-pepper noise, compression artifacts | Tweak OpenCV parameters |
| Silver | Wrong health value, missed enemy | Add more training data, adjust CNN |
| Gold | Wrong classification (win/lose) | Try a different classifier, tune hyperparams |

If you had one giant end-to-end model, a wrong prediction could be caused by
noise in the input, a missed enemy, or a bad classifier — and you would not
know which. Decomposing the problem lets you **isolate and fix** each failure
independently.

---

## 3. Bronze Layer: Pixel Cleanup

### What it does

Takes a raw screenshot and runs three deterministic OpenCV operations:

1. **Denoise (Non-Local Means)**: Replaces each pixel with a weighted average of
   similar-looking pixels across the whole image. This removes compression
   artifacts (blockiness from video codecs) while preserving edges better than
   a simple blur.

2. **Contrast Enhancement (CLAHE)**: Converts to LAB color space, applies
   adaptive histogram equalization to the L (lightness) channel, then converts
   back to RGB. Unlike a global histogram equalization, CLAHE operates on small
   tiles, so it does not amplify noise in large uniform regions (like sky or
   UI backgrounds).

3. **Sharpen (Unsharp Mask)**: Blurs the image, subtracts the blur from the
   original (this isolates the "edges"), then adds the edges back with extra
   weight. This reverses the slight softening introduced by the denoising step.

### The shape contract

Every function enforces `assert img.shape == (1440, 2560, 3)`. This is not
pedantry — it prevents **silent broadcasting bugs**. If a shape is wrong, a
later operation (like resizing a heatmap) would produce garbage without an error.

### Output

- A **clean PNG** (the preprocessed image)
- A **JSON metadata file** (mean brightness, std brightness, contrast)

The Silver layer reads the PNG. The metadata is informational only.

---

## 4. Silver Layer: Feature Extraction (The Core)

This is where the hardest work happens. The Silver layer must answer questions
like:

- How much health does the player have? (0.0 – 1.0)
- Where is the player? (cx, cy as fraction of screen)
- Where are the enemies? (up to 3, each with a bounding box)
- How much health does each enemy have?
- Is the player attacking? (binary)
- Is the player defending? (binary)
- Is the player taking damage? (binary)

### 4.1 Why not hardcode these?

A hardcoded approach would say: "The player health bar is at pixel coordinates
(2000, 20) to (2500, 50)". This works for one specific game at one specific
resolution — and breaks the moment either changes.

**The insight**: while the *position* of a health bar changes between games,
the *visual appearance* of a health bar (a filled rectangle whose fill ratio
indicates remaining health) is universal. A CNN can learn the *concept* of
"health bar" and then find it wherever it appears.

### 4.2 The Multi-Head Architecture

The SilverCNN uses a **shared backbone** with **specialized output heads**.
This is a specific instance of a broader family called "multi-task learning."

```
Input (3 × 1440 × 2560)
       │
       ▼
┌──────────────────┐
│   Shared Backbone │  ← 6 Conv → BN → ReLU blocks
│    (stride 64)    │    3→32→64→128→256→384→512 channels
│   23 × 40 feature │    1440→720→360→180→90→45→23
│       map         │    2560→1280→640→320→160→80→40
└──────┬───────────┘
       │
       ├─────────────────────┬──────────────────────┬──── ... ────┐
       ▼                     ▼                      ▼              ▼
┌──────────────┐   ┌──────────────┐   ┌─────────────────┐   ┌──────────┐
│ PlayerHealth │   │PlayerPosition│   │ EnemyHeatmap    │   │Attacking │
│  Linear(512→  │   │Linear(512→64)│   │ Conv2d(512→256) │   │  Linear  │
│   64→1) +    │   │   →64→2)     │   │   →ReLU→Conv2d  │   │512→32→1) │
│   Sigmoid    │   │  + Sigmoid   │   │   (256→3) +     │   │+ Sigmoid │
│  (scalar)    │   │  (cx, cy)    │   │   upscale→16×28 │   │(binary)  │
└──────────────┘   └──────────────┘   │   + Sigmoid     │   └──────────┘
                                      │  (3 heatmaps)   │
                                      └─────────────────┘
```

**Why share a backbone?** All tasks operate on the same input and benefit from
the same low-level features: edges, colors, textures. Sharing forces the network
to learn features that are generally useful rather than task-specific shortcuts.
It also means the model has far fewer total parameters than separate models for
each task.

### 4.3 The Heatmap Approach to Enemy Detection

Most object detectors (YOLO, Faster R-CNN) predict bounding boxes directly.
This project uses **heatmaps** instead. A heatmap is a low-resolution grid
(16 × 28 cells) where each cell value represents the probability of an enemy
being centered there.

**Why heatmaps?** Three reasons:

1. **Differentiability**: Heatmap prediction is just a per-pixel sigmoid with
   MSE loss. Backpropagation works trivially. Bounding box regression requires
   careful coordinate parametrization and can suffer from L1/L2 sensitivity.

2. **Uncertainty representation**: A heatmap can show "the enemy is somewhere in
   this region" as a soft blob. A bounding box must commit to crisp coordinates.

3. **No anchor boxes**: Heatmaps avoid the complexity of anchor box design,
   IoU computation, and NMS post-processing. The post-processing is a simple
   argmax over each channel.

**The cost**: Heatmaps lose precise localization. A 16×28 grid means each cell
covers about 90 × 90 pixels. The bounding boxes derived from heatmap peaks are
approximate. For this project's purpose (determining if the player is winning
or losing), approximate enemy positions are sufficient.

### 4.4 The Enemy-to-Health-Bar Relationship

Once the CNN identifies an enemy's position (via the heatmap peak), the health
bar location is derived geometrically: **always 30 pixels above the enemy's
center**. This is a domain-specific rule that holds for Jujutsu Shenanigans
(and many fighting games).

The CNN does not need to separately learn "find health bar" and "associate it
with enemy" — it learns "find enemy," and the code uses the known spatial
relationship. If a different game puts health bars elsewhere, this rule can be
adjusted without retraining the entire model.

### 4.5 Multi-Task Loss

The loss function is a sum of per-task losses:

```math
L = MSE(player_health, target)
  + MSE(player_position, target)
  + MSE(enemy_heatmap, target)
  + MSE(enemy_health, target)
  + BCE(attacking, target)
  + BCE(defending, target)
  + BCE(damage, target)
```

MSE is used for regression tasks (scalars, heatmaps) where the output is in
[0, 1]. BCE (binary cross-entropy) is used for binary classification tasks.

All losses are weighed equally. This is a deliberate choice to **not** introduce
hyperparameters that would need tuning. In practice, the scale of each loss
component differs (MSE for heatmaps is typically larger than BCE for binary
tasks), but the model adapts by allocating more capacity to the larger gradients.

### 4.6 Training Data

The SilverCNN is trained on **image-annotation pairs**. Each annotation is a
JSON file with:

```json
{
  "player_health": 0.75,
  "player_position": [0.5, 0.5],
  "enemies": [
    {"bbox": [100, 200, 60, 60], "health": 0.4,
     "health_bar_bbox": [100, 170, 50, 8], "confidence": 1.0}
  ],
  "attacking": true,
  "defending": false,
  "damage_indicator": false
}
```

**Synthetic data** is generated for pipeline testing. It renders simple colored
circles on a dark background — a green circle for the player (center), red
circles for enemies (random positions), and colored rectangles for health bars.
The annotations are generated deterministically alongside the images.

**Real data** will come from human-annotated screenshots. The annotation format
is the same; only the images differ.

### 4.7 MLflow Integration

Every training run is tracked with MLflow:

- **Parameters**: learning rate, batch size, epochs, model architecture details,
  dataset size
- **Metrics**: training and validation loss per epoch (logged at each step)
- **Artifacts**: the trained model (as a PyTorch MLflow model), sample
  predictions (text file comparing predicted vs target values)
- **Experiment organization**: runs are grouped by experiment name, making it
  easy to compare different hyperparameters

This allows you to answer: "Which learning rate produced the lowest validation
loss?" or "Did adding more training data improve the enemy health predictions?"

---

## 5. Gold Layer: State Classification

### How it uses Silver's output

The Gold layer ignores pixels entirely. It trains a classifier on the structured
`SilverFeatures` output:

| Input Feature | Why It Helps |
|---------------|-------------|
| `player_health` | Health context (a feature, not the label target) |
| `num_enemies` | Reads the pressure / numbers situation |
| `enemies[i].health` | Enemy gut health context |
| `attacking` | Aggression = the player is striking → `winning` |
| `defending` | Guards = reactive posture → part of neutral `stalemate` |
| `damage_indicator` | Flash / taking damage → `losing` |

The label tracks the player's **local initiative**, not health advantage:

- `winning`   = the player is striking the enemy
- `losing`    = the player is being hit (takes priority over attacking on overlap)
- `stalemate` = nothing is happening (neutral / defending)

The classifier (scikit-learn, XGBoost, or a small PyTorch MLP) produces one of
three labels: `winning`, `losing`, `stalemate`. Health stays in the feature
vector for the model to use if it helps, but it does not define the truth label.

### Why separate the classifier from the CNN?

The CNN extracts low-level features from pixels. The classifier makes a
high-level decision from those features. These are fundamentally different
problems:

- The CNN needs spatial understanding (where are the enemies?).
- The classifier needs relational understanding (is one enemy at low health
  while the player is at high health?).

A single model could do both, but debugging it would be harder. If the combined
model predicts "winning" when the player is actually losing, you would not know
if the CNN failed to find the enemy or the classifier failed to weigh the
evidence correctly.

---

## 6. Key Design Decisions (and Why)

### Decision 1: Fixed input size (1440 × 2560)

**Why**: Enables simple shape assertions that catch bugs early. The vast majority
of desktop screenshots are 16:9. If a non-standard resolution is used, the
Bronze layer silently resizes it rather than crashing.

**Trade-off**: Loss of detail if the input is upscaled, wasted computation if
downscaled.

### Decision 2: No data augmentation in SilverDataset

**Why**: The current synthetic data is already diverse (random enemy positions,
health values, attack states). Adding flips or rotations would break the
semantic meaning of "player on the left" (though a trained model might learn
to handle it). Real data augmentation will be added once real annotated data
is available.

### Decision 3: CPU inference by default

**Why**: The SilverCNN is small enough (~5M parameters) to run inference on a
CPU in under a second. This makes the pipeline accessible without a GPU.

### Decision 4: JSON as the data interchange format

**Why**: JSON is human-readable (you can open it in any text editor),
language-agnostic, and requires no additional libraries beyond the standard
library. For a few thousand annotations, JSON is fast enough. If the dataset
grows to hundreds of thousands of samples, migrating to a binary format (Parquet,
HDF5) would be the next step.

---

## 7. Common Failure Modes & How to Diagnose

| Symptom | Likely Cause | Check |
|---------|-------------|-------|
| Model always predicts 0.5 for player health | Loss not decreasing | Learning rate too high/low |
| Enemy heatmap is all zeros | No enemies in training data | Check annotation files |
| Validation loss lower than training | Data leakage or small dataset | Increase val_split |
| Bronze output looks washed out | CLAHE clip limit too low | Increase `clip_limit` parameter |
| Silver inference is slow on CPU | Model too large for real-time | Reduce backbone channels |

---

## 8. Data Requirements

For a **semi-viable model** (reliable on simple cases, may fail on edge cases):

| Task | Minimum Labeled Samples | Notes |
|------|------------------------|-------|
| Player health regression | 200–300 | Single scalar, visually salient |
| Player position | 100–200 | Trivial (always center) |
| Enemy detection | 500–1000 | 1–3 enemies, varying positions |
| Enemy health | 500–1000 | Tied to enemy detection |
| Attacking/defending | 300–500 each | Needs balanced classes |
| Damage indicator | 200–300 | Red flash is visually distinct |

**Total recommended**: **1000–2000 labeled screenshots** for a model that
handles common cases with ~75–85% reliability.

This estimate assumes:
- The game UI is visually consistent (no sudden art style changes)
- Annotations are accurate (human annotators or synthetic ground truth)
- The training/validation split captures the full distribution of game states

With fewer samples, the model may memorize patterns ("the health bar is always
green when health > 50%") rather than learning generalizable features. At
~5000+ samples, the model should approach its ceiling performance for this
architecture.

### Synthetic data strategy

Synthetic data can **augment** real data but cannot replace it entirely.
Synthetic frames are too clean — they lack the texture, lighting variation, and
visual noise of real game screenshots. A model trained purely on synthetic data
will perform poorly on real inputs (this is called the "sim-to-real gap").

**Recommended mix**: Start with ~500 synthetic samples to validate the pipeline,
then collect 1000+ real labeled screenshots for actual training.
