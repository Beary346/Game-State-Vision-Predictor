import json

import numpy as np
import pytest
import torch

from src.models.silver_cnn import SilverCNN, SilverDataset, SilverOutput
from src.pipeline.silver import (
    H,
    SilverFeatures,
    W,
    generate_synthetic_annotation,
    process_image,
    render_synthetic_frame,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def model() -> SilverCNN:
    return SilverCNN()


@pytest.fixture
def dummy_batch() -> SilverOutput:
    return SilverOutput(
        player_health=torch.tensor([0.75]),
        player_position=torch.tensor([[0.5, 0.5]]),
        enemy_heatmap=torch.zeros(1, 3, 16, 28),
        enemy_health=torch.tensor([[0.6, 0.0, 0.0]]),
        attacking=torch.tensor([0.1]),
        defending=torch.tensor([0.9]),
        damage_indicator=torch.tensor([0.0]),
    )


# ── Model forward pass ───────────────────────────────────────────────────────


class TestSilverCNN:
    def test_model_forward_pass(self, model):
        x = torch.randn(1, 3, H, W)
        out = model(x)
        assert isinstance(out, SilverOutput)
        assert out.player_health.shape == (1,)
        assert out.player_position.shape == (1, 2)
        assert out.enemy_heatmap.shape == (1, 3, 16, 28)
        assert out.enemy_health.shape == (1, 3)
        assert out.attacking.shape == (1,)
        assert out.defending.shape == (1,)
        assert out.damage_indicator.shape == (1,)

    def test_output_ranges(self, model):
        x = torch.randn(1, 3, H, W)
        out = model(x)
        assert 0.0 <= out.player_health.item() <= 1.0
        assert 0.0 <= out.attacking.item() <= 1.0
        assert 0.0 <= out.defending.item() <= 1.0
        assert 0.0 <= out.damage_indicator.item() <= 1.0
        assert torch.all(0.0 <= out.enemy_health) and torch.all(out.enemy_health <= 1.0)
        assert torch.all(0.0 <= out.player_position) and torch.all(out.player_position <= 1.0)

    def test_enemy_heatmap_sums_to_positive(self, model):
        x = torch.randn(1, 3, H, W)
        out = model(x)
        assert out.enemy_heatmap.sum() > 0.0

    def test_batch_independence(self, model):
        x = torch.randn(4, 3, H, W)
        out = model(x)
        assert out.player_health.shape == (4,)
        assert out.enemy_health.shape == (4, 3)
        assert out.attacking.shape == (4,)

    def test_model_is_trainable(self, model):
        x = torch.randn(2, 3, H, W)
        y = SilverOutput(
            player_health=torch.rand(2),
            player_position=torch.rand(2, 2),
            enemy_heatmap=torch.rand(2, 3, 16, 28),
            enemy_health=torch.rand(2, 3),
            attacking=torch.randint(0, 2, (2,)).float(),
            defending=torch.randint(0, 2, (2,)).float(),
            damage_indicator=torch.randint(0, 2, (2,)).float(),
        )
        out = model(x)
        loss = (
            torch.nn.functional.mse_loss(out.player_health, y.player_health)
            + torch.nn.functional.mse_loss(out.player_position, y.player_position)
            + torch.nn.functional.mse_loss(out.enemy_heatmap, y.enemy_heatmap)
            + torch.nn.functional.mse_loss(out.enemy_health, y.enemy_health)
            + torch.nn.functional.binary_cross_entropy(out.attacking, y.attacking)
            + torch.nn.functional.binary_cross_entropy(out.defending, y.defending)
            + torch.nn.functional.binary_cross_entropy(out.damage_indicator, y.damage_indicator)
        )
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.abs().sum() > 0
                break


# ── SilverOutput dataclass ────────────────────────────────────────────────────


class TestSilverOutput:
    def test_to_dict_round_trip(self, dummy_batch):
        d = dummy_batch._asdict()
        assert "player_health" in d
        assert "player_position" in d
        assert "enemy_heatmap" in d
        assert "enemy_health" in d
        assert "attacking" in d
        assert "defending" in d
        assert "damage_indicator" in d

    def test_device_transfer(self, dummy_batch):
        cpu_tensors = [v.device.type == "cpu" for v in dummy_batch]
        assert all(cpu_tensors)

    def test_enemy_health_shape(self, dummy_batch):
        assert dummy_batch.enemy_health.shape[-1] == 3


# ── SilverFeatures ────────────────────────────────────────────────────────────


class TestSilverFeatures:
    def test_dataclass_fields(self):
        sf = SilverFeatures(
            image_path="/fake/path.png",
            player_health=0.5,
            player_position=(0.5, 0.5),
            enemies=[],
            attacking=False,
            defending=True,
            damage_indicator=False,
            num_enemies=0,
        )
        assert sf.player_health == 0.5
        assert sf.defending is True
        assert sf.num_enemies == 0

    def test_enemy_entry_schema(self):
        enemy = {"bbox": (100, 200, 50, 80), "health": 0.75, "health_bar_bbox": (100, 180, 50, 10), "confidence": 0.92}
        sf = SilverFeatures(
            image_path="/p.png",
            player_health=0.8,
            player_position=(0.5, 0.5),
            enemies=[enemy],
            attacking=True,
            defending=False,
            damage_indicator=False,
            num_enemies=1,
        )
        assert len(sf.enemies) == 1
        assert sf.enemies[0]["health"] == 0.75
        assert sf.enemies[0]["confidence"] == 0.92

    def test_json_serializable(self):
        sf = SilverFeatures(
            image_path="/p.png",
            player_health=0.5,
            player_position=(0.5, 0.5),
            enemies=[{"bbox": (1, 2, 3, 4), "health": 0.6, "health_bar_bbox": (1, 0, 3, 1), "confidence": 0.9}],
            attacking=False,
            defending=True,
            damage_indicator=False,
            num_enemies=1,
        )
        d = sf._asdict() if hasattr(sf, "_asdict") else sf.__dict__
        js = json.dumps(d)
        assert "player_health" in js


# ── Synthetic data generation ─────────────────────────────────────────────────


class TestSyntheticData:
    def test_render_synthetic_frame_shape(self):
        img = render_synthetic_frame()
        assert img.shape == (H, W, 3)
        assert img.dtype == np.uint8

    def test_render_synthetic_has_player_center(self):
        img = render_synthetic_frame()
        center_color = img[H // 2, W // 2]
        assert not np.all(center_color == 0)

    def test_generate_synthetic_annotation_returns_expected_keys(self):
        ann = generate_synthetic_annotation()
        assert "player_health" in ann
        assert "player_position" in ann
        assert "enemies" in ann
        assert "attacking" in ann
        assert "defending" in ann
        assert "damage_indicator" in ann

    def test_synthetic_output_ranges(self):
        ann = generate_synthetic_annotation()
        assert 0.0 <= ann["player_health"] <= 1.0
        assert all(0.0 <= v <= 1.0 for v in ann["player_position"])
        for enemy in ann["enemies"]:
            assert 0.0 <= enemy["health"] <= 1.0
            assert len(enemy["bbox"]) == 4


# ── SilverDataset ─────────────────────────────────────────────────────────────


class TestSilverDataset:
    def test_dataset_len(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "annotations").mkdir()
        img = render_synthetic_frame()
        cv2_import = pytest.importorskip("cv2")
        cv2_import.imwrite(str(tmp_path / "images" / "frame_000.png"), cv2_import.cvtColor(img, cv2_import.COLOR_RGB2BGR))
        with open(tmp_path / "annotations" / "frame_000.json", "w") as f:
            json.dump(generate_synthetic_annotation(), f)
        ds = SilverDataset(str(tmp_path), transforms=None)
        assert len(ds) == 1

    def test_dataset_getitem(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "annotations").mkdir()
        img = render_synthetic_frame()
        cv2_import = pytest.importorskip("cv2")
        cv2_import.imwrite(str(tmp_path / "images" / "frame_000.png"), cv2_import.cvtColor(img, cv2_import.COLOR_RGB2BGR))
        ann = generate_synthetic_annotation()
        with open(tmp_path / "annotations" / "frame_000.json", "w") as f:
            json.dump(ann, f)
        ds = SilverDataset(str(tmp_path), transforms=None)
        x, y = ds[0]
        assert x.shape == (3, H, W)
        assert isinstance(y, SilverOutput)
        assert y.player_health.item() == pytest.approx(ann["player_health"], abs=1e-6)


# ── Silver pipeline ──────────────────────────────────────────────────────────


class TestSilverPipeline:
    def test_process_image_returns_features(self, tmp_path, test_screenshot_path):
        result = process_image(test_screenshot_path, str(tmp_path))
        assert isinstance(result, dict)
        assert "silver_features" in result
        assert "features_json" in result

    def test_process_image_json_output(self, tmp_path, test_screenshot_path):
        result = process_image(test_screenshot_path, str(tmp_path))
        with open(result["features_json"]) as f:
            data = json.load(f)
        assert "player_health" in data
        assert "player_position" in data
        assert "enemies" in data
        assert "attacking" in data
        assert "defending" in data
        assert "damage_indicator" in data
        assert "num_enemies" in data

    def test_process_image_with_model(self, tmp_path, test_screenshot_path):
        cnn = SilverCNN()
        result = process_image(test_screenshot_path, str(tmp_path), model=cnn)
        assert "silver_features" in result

    def test_process_image_json_valid(self, tmp_path, test_screenshot_path):
        result = process_image(test_screenshot_path, str(tmp_path))
        with open(result["features_json"]) as f:
            data = json.load(f)
        sf = SilverFeatures(**data)
        assert isinstance(sf.player_health, (float, int))
