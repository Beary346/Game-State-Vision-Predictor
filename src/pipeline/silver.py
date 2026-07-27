import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from src.models.silver_cnn import SilverCNN, SilverOutput

H = 1440
W = 2560


@dataclass
class SilverFeatures:
    image_path: str
    player_health: float
    player_position: tuple[float, float]
    enemies: list[dict]
    attacking: bool
    defending: bool
    damage_indicator: bool
    num_enemies: int


@dataclass
class EnemyAnnotation:
    bbox: tuple[int, int, int, int]
    health: float
    health_bar_bbox: tuple[int, int, int, int]
    confidence: float


def _extract_enemies_from_heatmap(
    heatmap: np.ndarray,
    confidence_threshold: float = 0.3,
    max_enemies: int = 3,
) -> list[EnemyAnnotation]:
    enemies: list[EnemyAnnotation] = []
    for i in range(heatmap.shape[0]):
        hmap = heatmap[i]
        max_val = hmap.max()
        if max_val < confidence_threshold:
            continue
        max_idx = hmap.argmax()
        hy, hx = divmod(int(max_idx), hmap.shape[1])
        cx = int(hx * W / hmap.shape[1])
        cy = int(hy * H / hmap.shape[0])
        w, h = 60, 100
        bbox = (cx - w // 2, cy - h // 2, w, h)
        health_bar = (cx - 25, cy - 30, 50, 8)
        enemy_health = float(np.random.uniform(0.3, 1.0))
        enemies.append(
            EnemyAnnotation(
                bbox=bbox,
                health=enemy_health,
                health_bar_bbox=health_bar,
                confidence=float(max_val),
            )
        )
    return enemies[:max_enemies]


def _model_to_features(output: SilverOutput, image_path: str) -> SilverFeatures:
    heatmap_np = output.enemy_heatmap.detach().squeeze(0).cpu().numpy()
    enemies_raw = _extract_enemies_from_heatmap(heatmap_np)
    enemies_dict = [asdict(e) for e in enemies_raw]

    return SilverFeatures(
        image_path=image_path,
        player_health=float(output.player_health.detach().cpu().item()),
        player_position=tuple(output.player_position.detach().cpu().squeeze(0).tolist()),
        enemies=enemies_dict,
        attacking=float(output.attacking.detach().cpu().item()) > 0.5,
        defending=float(output.defending.detach().cpu().item()) > 0.5,
        damage_indicator=float(output.damage_indicator.detach().cpu().item()) > 0.5,
        num_enemies=len(enemies_dict),
    )


def _default_features(image_path: str) -> SilverFeatures:
    return SilverFeatures(
        image_path=image_path,
        player_health=0.0,
        player_position=(0.5, 0.5),
        enemies=[],
        attacking=False,
        defending=False,
        damage_indicator=False,
        num_enemies=0,
    )


def render_synthetic_frame(
    num_enemies: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (50, 50, 60)

    cv2.circle(img, (W // 2, H // 2), 40, (0, 200, 50), -1)

    if num_enemies is None:
        num_enemies = random.randint(1, 3)

    for _ in range(num_enemies):
        ex = random.randint(100, W - 100)
        ey = random.randint(100, H - 100)
        cv2.circle(img, (ex, ey), 35, (200, 50, 50), -1)
        health_w = 50
        health_h = 8
        health_x = ex - health_w // 2
        health_y = ey - 40
        health_pct = random.uniform(0.2, 1.0)
        cv2.rectangle(img, (health_x, health_y), (health_x + health_w, health_y + health_h), (80, 80, 80), -1)
        fill_w = int(health_w * health_pct)
        bar_color = (0, 200, 0) if health_pct > 0.5 else (0, 0, 200)
        cv2.rectangle(img, (health_x, health_y), (health_x + fill_w, health_y + health_h), bar_color, -1)

    cv2.rectangle(img, (W - 520, 20), (W - 20, 50), (80, 80, 80), -1)
    player_health = random.uniform(0.3, 1.0)
    fill_w = int(500 * player_health)
    bar_color = (0, 200, 0) if player_health > 0.5 else (0, 200, 200)
    cv2.rectangle(img, (W - 520, 20), (W - 520 + fill_w, 50), bar_color, -1)

    return img


def generate_synthetic_annotation(
    num_enemies: int | None = None,
    seed: int | None = None,
) -> dict:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if num_enemies is None:
        num_enemies = random.randint(1, 3)

    player_health = random.uniform(0.3, 1.0)
    enemies: list[dict] = []
    for _ in range(num_enemies):
        ex = random.randint(100, W - 100)
        ey = random.randint(100, H - 100)
        enemy_health = random.uniform(0.2, 1.0)
        enemies.append(
            {
                "bbox": (ex - 30, ey - 30, 60, 60),
                "health": enemy_health,
                "health_bar_bbox": (ex - 25, ey - 40, 50, 8),
                "confidence": 1.0,
            }
        )

    return {
        "player_health": player_health,
        "player_position": [0.5, 0.5],
        "enemies": enemies,
        "attacking": random.choice([True, False]),
        "defending": random.choice([True, False]),
        "damage_indicator": random.random() < 0.2,
    }


def process_image(
    image_path: str,
    output_dir: str,
    model: SilverCNN | None = None,
) -> dict:
    if model is not None:
        model.eval()
        img_bgr = cv2.imread(image_path)
        assert img_bgr is not None, f"Could not read image at {image_path}"
        h, w = img_bgr.shape[:2]
        if (h, w) != (H, W):
            img_bgr = cv2.resize(img_bgr, (W, H))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
        features = _model_to_features(output, image_path)
    else:
        features = _default_features(image_path)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    json_path = str(out_dir / f"{stem}_silver.json")

    with open(json_path, "w") as f:
        json.dump(asdict(features), f, indent=2)

    return {"silver_features": asdict(features), "features_json": json_path}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Silver: EDA / feature extraction")
    ap.add_argument("-i", "--image", required=True, help="path to preprocessed Bronze image")
    ap.add_argument("-o", "--output", default="data/silver", help="output directory")
    ap.add_argument("--model", default=None, help="path to SilverCNN checkpoint (optional)")
    args = vars(ap.parse_args())

    model = None
    if args["model"]:
        cnn = SilverCNN()
        cnn.load_state_dict(torch.load(args["model"], map_location="cpu"))
        model = cnn

    result = process_image(args["image"], args["output"], model=model)
    print(f"Silver features : {result['features_json']}")
    print(f"Player health   : {result['silver_features']['player_health']:.3f}")
    print(f"Enemies detected: {result['silver_features']['num_enemies']}")
    print("Silver analysis complete.")
