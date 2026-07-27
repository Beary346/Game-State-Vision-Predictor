import json
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

H = 1440
W = 2560


class SilverOutput(NamedTuple):
    player_health: torch.Tensor
    player_position: torch.Tensor
    enemy_heatmap: torch.Tensor
    enemy_health: torch.Tensor
    attacking: torch.Tensor
    defending: torch.Tensor
    damage_indicator: torch.Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, padding: int | None = None):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SilverCNN(nn.Module):
    """Multi-head CNN that extracts game-relevant features from a full-resolution screenshot.

    Backbone: 6 downsampling blocks → feature map at stride 64 (≈22×40).
    Heads:
    - player_health:    scalar regression (0..1)
    - player_position:  (cx, cy) normalized (0..1)
    - enemy_heatmap:    3 heatmaps at 16×28 (one per enemy slot)
    - enemy_health:     3 scalars (0..1)
    - attacking:        binary (0..1)
    - defending:        binary (0..1)
    - damage_indicator: binary (0..1)
    """

    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            ConvBlock(3, 32, kernel=7, stride=4, padding=3),   # → 360×640
            ConvBlock(32, 64, kernel=3, stride=2),               # → 180×320
            ConvBlock(64, 128, kernel=3, stride=2),              # → 90×160
            ConvBlock(128, 256, kernel=3, stride=2),             # → 45×80
            ConvBlock(256, 384, kernel=3, stride=2),             # → 23×40
            ConvBlock(384, 512, kernel=3, stride=1),             # → 23×40
        )

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, kernel_size=1),
        )

        self.enemy_health_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
            nn.Sigmoid(),
        )

        self.player_health_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.player_position_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

        self.attacking_head = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.defending_head = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.damage_head = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> SilverOutput:
        features = self.backbone(x)
        pooled = features.mean(dim=[2, 3])

        enemy_heatmap = F.interpolate(self.heatmap_head(features), size=(16, 28), mode="bilinear", align_corners=False)
        enemy_heatmap = torch.sigmoid(enemy_heatmap)

        enemy_health = self.enemy_health_head(pooled)
        player_health = self.player_health_head(pooled).squeeze(-1)
        player_position = self.player_position_head(pooled)
        attacking = self.attacking_head(pooled).squeeze(-1)
        defending = self.defending_head(pooled).squeeze(-1)
        damage_indicator = self.damage_head(pooled).squeeze(-1)

        return SilverOutput(
            player_health=player_health,
            player_position=player_position,
            enemy_heatmap=enemy_heatmap,
            enemy_health=enemy_health,
            attacking=attacking,
            defending=defending,
            damage_indicator=damage_indicator,
        )


class SilverDataset(Dataset):
    """Load images and JSON annotations for training the SilverCNN.

    Expects the following directory layout::

        root/
            images/       # PNG screenshots (BGR or RGB uint8)
            annotations/  # JSON files matching each image (same stem)
    """

    def __init__(self, root: str, transforms=None):
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.annot_dir = self.root / "annotations"
        self.transforms = transforms

        self.samples: list[tuple[Path, Path]] = []
        for img_path in sorted(self.image_dir.glob("*.png")):
            stem = img_path.stem
            ann_path = self.annot_dir / f"{stem}.json"
            if ann_path.exists():
                self.samples.append((img_path, ann_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, SilverOutput]:
        img_path, ann_path = self.samples[idx]

        import cv2
        import numpy as np

        img_bgr = cv2.imread(str(img_path))
        assert img_bgr is not None, f"Failed to load {img_path}"
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)

        with open(ann_path) as f:
            ann = json.load(f)

        target = self._annotation_to_output(ann)
        return img_tensor, target

    @staticmethod
    def _annotation_to_output(ann: dict) -> SilverOutput:
        enemy_heatmap = torch.zeros(3, 16, 28)
        enemy_health = torch.zeros(3)
        for i, enemy in enumerate(ann.get("enemies", [])[:3]):
            cx, cy = enemy["bbox"][0] + enemy["bbox"][2] / 2, enemy["bbox"][1] + enemy["bbox"][3] / 2
            hx = int(cx * 16 / W)
            hy = int(cy * 16 / H)
            hx = min(hx, 27)
            hy = min(hy, 15)
            enemy_heatmap[i, hy, hx] = 1.0
            enemy_health[i] = enemy["health"]

        return SilverOutput(
            player_health=torch.tensor(ann["player_health"], dtype=torch.float32),
            player_position=torch.tensor(ann["player_position"], dtype=torch.float32),
            enemy_heatmap=enemy_heatmap,
            enemy_health=enemy_health,
            attacking=torch.tensor(float(ann["attacking"]), dtype=torch.float32),
            defending=torch.tensor(float(ann["defending"]), dtype=torch.float32),
            damage_indicator=torch.tensor(float(ann["damage_indicator"]), dtype=torch.float32),
        )


def compute_loss(pred: SilverOutput, target: SilverOutput) -> torch.Tensor:
    return (
        F.mse_loss(pred.player_health, target.player_health)
        + F.mse_loss(pred.player_position, target.player_position)
        + F.mse_loss(pred.enemy_heatmap, target.enemy_heatmap)
        + F.mse_loss(pred.enemy_health, target.enemy_health)
        + F.binary_cross_entropy(pred.attacking, target.attacking)
        + F.binary_cross_entropy(pred.defending, target.defending)
        + F.binary_cross_entropy(pred.damage_indicator, target.damage_indicator)
    )
