import json
import tempfile
from pathlib import Path

from src.pipeline.silver_train import generate_and_train, generate_synthetic_dataset


class TestGenerateSyntheticDataset:
    def test_creates_images_and_annotations(self):
        tmp = tempfile.mkdtemp()
        path = generate_synthetic_dataset(tmp, num_samples=5, seed=0)
        img_dir = Path(path) / "images"
        ann_dir = Path(path) / "annotations"
        assert img_dir.exists()
        assert ann_dir.exists()
        pngs = sorted(img_dir.glob("*.png"))
        jsons = sorted(ann_dir.glob("*.json"))
        assert len(pngs) == 5
        assert len(jsons) == 5
        for png, js in zip(pngs, jsons):
            assert png.stem == js.stem

    def test_annotations_match_synthetic_format(self):
        tmp = tempfile.mkdtemp()
        path = generate_synthetic_dataset(tmp, num_samples=1, seed=0)
        ann_file = next(Path(path).glob("annotations/*.json"))
        with open(ann_file) as f:
            ann = json.load(f)
        assert "player_health" in ann
        assert "player_position" in ann
        assert "enemies" in ann
        assert "attacking" in ann
        assert "defending" in ann
        assert "damage_indicator" in ann


class TestGenerateAndTrain:
    def test_training_completes(self):
        result = generate_and_train(
            synthetic_samples=10,
            learning_rate=1e-3,
            batch_size=2,
            num_epochs=2,
            val_split=0.2,
            experiment_name="test_silver_train",
            run_name="pytest_run",
            device="cpu",
        )
        assert "run_id" in result
        assert "best_val_loss" in result
        assert isinstance(result["best_val_loss"], float)
        assert result["best_val_loss"] >= 0.0

    def test_different_params_produce_different_runs(self):
        r1 = generate_and_train(
            synthetic_samples=10,
            learning_rate=1e-2,
            batch_size=2,
            num_epochs=2,
            val_split=0.2,
            experiment_name="test_silver_train_diff",
            run_name="run_lr_1e2",
            device="cpu",
        )
        r2 = generate_and_train(
            synthetic_samples=10,
            learning_rate=1e-4,
            batch_size=2,
            num_epochs=2,
            val_split=0.2,
            experiment_name="test_silver_train_diff",
            run_name="run_lr_1e4",
            device="cpu",
        )
        assert r1["run_id"] != r2["run_id"]
