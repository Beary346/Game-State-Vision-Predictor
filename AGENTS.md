# Game State Vision Predictor

## Objective
Detect and predict "game state" combinations from fighting game screenshots.

## Inputs
Raw game screenshots.

## Outputs
Machine output categorized as `winning`, `losing`, or `stalemate`. Classification is based on:
- **Health ratio** (player vs enemy)
- **Aggression** (is the player attacking?)
- **Defense** (how is the player defending?)

## Pipeline

### Bronze (Raw Screenshot)
OpenCV preprocessing.

### Silver (Clean Data)
EDA cleaning — separates Bronze image into manageable chunks for noise removal.

### Gold (Trained Model)
MLflow tests a wide variety of models to find the best fit. Trains a model using plain-English category labels derived from Silver data.

## Constraints
- **Tensor shape assertions**: enforce `assert img.shape == (H, W, 3)` to prevent silent broadcasting bugs. H=2560, W=1440 (desktop resolution) to ensure all data is captured.
- **Large dataset**: substantial training data needed for the model to learn which features to extract from Bronze→Silver.
- **One Game structure**: train on a single game initially to avoid conflicting noise. Game: **Jujutsu Shenanigans (Roblox)** — chosen for its high attack variation and familiarity.

## Core Features
- OpenCV for data extraction
- Effective noise removal (distinguish important data from noise)
- Final state classification (winning/losing/stalemate) from Gold data
- Player identification: player is always center screen, has no name above their head, and no health bar on their head (this distinguishes them from enemies)

## Stack

| Category       | Tool           | Purpose                      |
|----------------|----------------|------------------------------|
| Assistance     | OpenCode       | Coding assistance            |
| Modeling       | Scikit-learn   | Models / Train-Test Data     |
| Math           | NumPy          | Math operations              |
| Data           | Pandas         | Data manipulation            |
| Workflow       | MLflow         | ML workflow organization     |
| Image          | OpenCV         | Image processing             |
| Visualization  | JupyterLab     | Data visualization           |
| Visualization  | Matplotlib     | Data visualization           |
| Visualization  | Seaborn        | Statistical visualization    |
| Modeling       | PyTorch        | Deep learning models         |
| Modeling       | XGBoost        | Gradient boosting models     |

## Rules
- Write tests before implementation.
- Write one document at a time and wait for my confirmation.
- Add comments on your code, prioritizing developer readability: concise docstrings, and descriptive variable names can replace the need for excessive inline comments.
- Don't just log numbers: log correlation matrices, UMAP/t-SNE embeddings, and confusion matrices. Take advantage of jupyterlab, A Web UI (or notebook) for interactive, human-in-the-loop use. 
