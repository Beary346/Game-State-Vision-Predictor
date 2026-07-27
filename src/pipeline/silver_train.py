import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch.utils.data import DataLoader, random_split

from src.models.silver_cnn import SilverCNN, SilverDataset, compute_loss
from src.pipeline.silver import generate_synthetic_annotation, render_synthetic_frame


def _ensure_tracking_uri():
    """Set a safe MLflow tracking URI (SQLite) that won't conflict with spaces in paths."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    safe_path = os.path.join(tempfile.gettempdir(), "mlruns", "mlflow.db")
    os.makedirs(os.path.dirname(safe_path), exist_ok=True)
    uri = f"sqlite:///{safe_path}"
    mlflow.set_tracking_uri(uri)
    return uri

H = 1440
W = 2560


def generate_synthetic_dataset(output_dir: str, num_samples: int = 100, seed: int = 0) -> str:
    """Generate a synthetic labeled dataset for pipeline testing.

    Creates ``images/`` and ``annotations/`` subdirectories under *output_dir*.
    Returns the path to *output_dir*.
    """
    out = Path(output_dir)
    img_dir = out / "images"
    ann_dir = out / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)

    for i in range(num_samples):
        num_enemies = int(rng.integers(1, 4))
        frame = render_synthetic_frame(num_enemies=num_enemies, seed=seed + i)
        ann = generate_synthetic_annotation(num_enemies=num_enemies, seed=seed + i)
        cv2.imwrite(str(img_dir / f"frame_{i:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        with open(ann_dir / f"frame_{i:04d}.json", "w") as f:
            json.dump(ann, f)

    return str(out)


def _log_sample_predictions(model, loader, device, num_samples: int = 4):
    """Log sample predictions vs ground truth as a text artifact."""
    model.eval()
    rows = ["sample|task|predicted|target"]
    count = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            preds = model(images)
            for b in range(images.size(0)):
                rows.append(f"{count}|player_health|{preds.player_health[b].item():.4f}|{targets.player_health[b].item():.4f}")
                rows.append(f"{count}|attacking|{preds.attacking[b].item():.4f}|{targets.attacking[b].item():.4f}")
                rows.append(f"{count}|defending|{preds.defending[b].item():.4f}|{targets.defending[b].item():.4f}")
                rows.append(f"{count}|damage|{preds.damage_indicator[b].item():.4f}|{targets.damage_indicator[b].item():.4f}")
                rows.append(f"{count}|enemy_health_0|{preds.enemy_health[b, 0].item():.4f}|{targets.enemy_health[b, 0].item():.4f}")
                count += 1
                if count >= num_samples:
                    break
            if count >= num_samples:
                break
    mlflow.log_text("\n".join(rows), "sample_predictions.txt")


def train_silver_model(
    data_root: str,
    learning_rate: float = 1e-4,
    batch_size: int = 2,
    num_epochs: int = 50,
    val_split: float = 0.15,
    experiment_name: str = "silver_cnn",
    run_name: str | None = None,
    seed: int = 42,
    device: str = "auto",
):
    """Train a SilverCNN with full MLflow tracking.

    Parameters
    ----------
    data_root : str
        Path to dataset directory containing ``images/`` and ``annotations/``.
    learning_rate : float
        Adam learning rate.
    batch_size : int
        Per-GPU batch size.
    num_epochs : int
        Number of full passes over the training set.
    val_split : float
        Fraction of data held out for validation (0.0 to skip).
    experiment_name : str
        MLflow experiment name.
    run_name : str | None
        MLflow run name (auto-generated if ``None``).
    seed : int
        Random seed for reproducibility.
    device : str
        ``"auto"`` detects CUDA; otherwise pass ``"cpu"`` or ``"cuda"``.
    """
    torch.manual_seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _ensure_tracking_uri()

    dataset = SilverDataset(data_root)
    if len(dataset) == 0:
        raise RuntimeError(f"No valid image/annotation pairs found in {data_root}")

    if val_split > 0.0:
        val_len = int(len(dataset) * val_split)
        train_len = len(dataset) - val_len
        train_ds, val_ds = random_split(dataset, [train_len, val_len])
    else:
        train_ds = dataset
        val_ds = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None

    model = SilverCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if run_name is None:
        run_name = f"silver_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "val_split": val_split,
            "seed": seed,
            "device": device,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds) if val_ds else 0,
            "model": "SilverCNN",
            "backbone_blocks": 6,
            "backbone_channels": [32, 64, 128, 256, 384, 512],
        })

        best_val_loss = float("inf")

        for epoch in range(1, num_epochs + 1):
            model.train()
            train_loss = 0.0
            train_count = 0

            for images, targets in train_loader:
                images = images.to(device)
                targets = targets._replace(
                    player_health=targets.player_health.to(device),
                    player_position=targets.player_position.to(device),
                    enemy_heatmap=targets.enemy_heatmap.to(device),
                    enemy_health=targets.enemy_health.to(device),
                    attacking=targets.attacking.to(device),
                    defending=targets.defending.to(device),
                    damage_indicator=targets.damage_indicator.to(device),
                )

                preds = model(images)
                loss = compute_loss(preds, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                train_count += images.size(0)

            train_loss /= train_count

            metrics = {"train_loss": train_loss}

            if val_loader:
                model.eval()
                val_loss = 0.0
                val_count = 0
                with torch.no_grad():
                    for images, targets in val_loader:
                        images = images.to(device)
                        targets = targets._replace(
                            player_health=targets.player_health.to(device),
                            player_position=targets.player_position.to(device),
                            enemy_heatmap=targets.enemy_heatmap.to(device),
                            enemy_health=targets.enemy_health.to(device),
                            attacking=targets.attacking.to(device),
                            defending=targets.defending.to(device),
                            damage_indicator=targets.damage_indicator.to(device),
                        )
                        preds = model(images)
                        loss = compute_loss(preds, targets)
                        val_loss += loss.item() * images.size(0)
                        val_count += images.size(0)
                val_loss /= val_count
                metrics["val_loss"] = val_loss

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
                        best_path = tmp.name
                        torch.save(model.state_dict(), best_path)
                    mlflow.log_artifact(best_path, artifact_path="checkpoints")
                    Path(best_path).unlink(missing_ok=True)

            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch {epoch:3d}/{num_epochs}  |  "
                      f"train_loss: {train_loss:.4f}  |  "
                      f"{'val_loss: ' + f'{val_loss:.4f}' if val_loader else '(no val)'}")

            mlflow.log_metrics(metrics, step=epoch)

        mlflow.pytorch.log_model(model, "model", serialization_format="pickle")
        logged_model_path = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.log_param("logged_model", logged_model_path)

        if val_loader:
            _log_sample_predictions(model, val_loader, device)
        else:
            _log_sample_predictions(model, train_loader, device)

        print(f"\nRun {mlflow.active_run().info.run_id} complete.")
        print(f"Best val_loss: {best_val_loss:.4f}")
        print(f"Model logged to: {logged_model_path}")

        return {"run_id": mlflow.active_run().info.run_id, "best_val_loss": best_val_loss}


def generate_and_train(
    synthetic_samples: int = 200,
    learning_rate: float = 1e-4,
    batch_size: int = 2,
    num_epochs: int = 50,
    val_split: float = 0.15,
    experiment_name: str = "silver_cnn",
    run_name: str | None = None,
    device: str = "auto",
):
    """Generate a synthetic dataset and train the SilverCNN in one step.

    This is the recommended entry point for first-time users.
    """
    import tempfile

    tmp = tempfile.mkdtemp()
    print(f"Generating {synthetic_samples} synthetic samples in {tmp}...")
    generate_synthetic_dataset(tmp, num_samples=synthetic_samples)

    return train_silver_model(
        data_root=tmp,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        val_split=val_split,
        experiment_name=experiment_name,
        run_name=run_name,
        device=device,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SilverCNN Training with MLflow")
    ap.add_argument("--data-root", default=None, help="path to dataset (images/ + annotations/)")
    ap.add_argument("--synthetic", type=int, default=200, help="generate N synthetic samples if --data-root not given")
    ap.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    ap.add_argument("--batch-size", type=int, default=2, help="batch size")
    ap.add_argument("--epochs", type=int, default=50, help="number of epochs")
    ap.add_argument("--val-split", type=float, default=0.15, help="validation split fraction")
    ap.add_argument("--experiment", default="silver_cnn", help="MLflow experiment name")
    ap.add_argument("--run-name", default=None, help="MLflow run name")
    ap.add_argument("--device", default="auto", help='device: "auto", "cpu", or "cuda"')
    args = vars(ap.parse_args())

    if args["data_root"]:
        train_silver_model(
            data_root=args["data_root"],
            learning_rate=args["lr"],
            batch_size=args["batch_size"],
            num_epochs=args["epochs"],
            val_split=args["val_split"],
            experiment_name=args["experiment"],
            run_name=args["run_name"],
            device=args["device"],
        )
    else:
        generate_and_train(
            synthetic_samples=args["synthetic"],
            learning_rate=args["lr"],
            batch_size=args["batch_size"],
            num_epochs=args["epochs"],
            val_split=args["val_split"],
            experiment_name=args["experiment"],
            run_name=args["run_name"],
            device=args["device"],
        )
