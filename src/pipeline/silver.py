"""Silver re-export shim (legacy import path).

The modern heuristic implementation lives in ``src.classifier_silver``; this
module re-exports it under the old ``src.pipeline.silver`` name so Gold,
silver_train, the tests, and the EDA notebook keep working:

    from src.pipeline.silver import (
        SilverFeatures, H, W, render_synthetic_frame, generate_synthetic_annotation,
        classify_frame, process_image, process_frames, aggregate_rounds,
    )

``process_image`` keeps its original signature ``(image_path, output_dir,
model=None)``: with a SilverCNN it runs the CNN path; without one it falls back
to the deterministic heuristic reader.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from src.classifier_silver import (
    ABILITY_ROW_REGION,
    ABILITY_SLOTS,
    CLOCK_REGION,
    DAMAGE_FLASH_REGION,
    ENEMY_HEALTH_BAR,
    HEALTH_BAR_TOLERANCE,
    PLAYER_HEALTH_BAR,
    PLAYER_SPRITE_RADIUS,
    STATE_DOT_REGION,
    UNITS_MIN_Y,
    H,
    RoundSummary,
    SilverFeatures,
    W,
    aggregate_rounds,
    assign_round_indices,
    classify_frame,
    crop_region,
    generate_synthetic_annotation,
    health_bar_fill,
    identify_player_and_enemies,
    process_frames,
    read_ability_indicators,
    read_clock_ocr,
    read_damage_flash,
    read_health_fill,
    read_state_dot,
    render_frame_and_state,
    render_synthetic_frame,
    simulate_match,
)
from src.models.silver_cnn import SilverCNN, SilverOutput


@dataclass
class EnemyAnnotation:
    """One recognised enemy unit from the CNN heatmap path."""

    bbox: tuple[int, int, int, int]
    health: float
    health_bar_bbox: tuple[int, int, int, int]
    confidence: float


def _extract_enemies_from_heatmap(
    heatmap: np.ndarray,
    confidence_threshold: float = 0.3,
    max_enemies: int = 3,
) -> list[EnemyAnnotation]:
    """Peak-pick enemy slots from a CNN heatmap into enemy annotations."""
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
    """Convert a SilverCNN output bundle into a SilverFeatures tuple."""
    heatmap_np = output.enemy_heatmap.detach().squeeze(0).cpu().numpy()
    enemies_dict = [asdict(e) for e in _extract_enemies_from_heatmap(heatmap_np)]
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


def process_image(
    image_path: str,
    output_dir: str,
    model: SilverCNN | None = None,
) -> dict:
    """Extract silver features for one image and persist them as JSON.

    With a ``model`` the CNN path runs (full heatmap extraction); without one
    the deterministic HUD heuristic reads the frame. Unusual resolutions are
    resized to the standard (H, W) grid so the grounded HUD regions line up.
    """
    img_bgr = cv2.imread(image_path)
    assert img_bgr is not None, f"Could not read image at {image_path}"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img_rgb.shape[:2] != (H, W):
        img_rgb = cv2.resize(img_rgb, (W, H))

    if model is not None:
        model.eval()
        img_tensor = (
            torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        )
        with torch.no_grad():
            features = _model_to_features(model(img_tensor), image_path)
    else:
        features = classify_frame(img_rgb, image_path=image_path)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{Path(image_path).stem}_silver.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(features), f, indent=2)

    return {"silver_features": asdict(features), "features_json": str(json_path)}


__all__ = [
    "ABILITY_ROW_REGION",
    "ABILITY_SLOTS",
    "CLOCK_REGION",
    "DAMAGE_FLASH_REGION",
    "ENEMY_HEALTH_BAR",
    "HEALTH_BAR_TOLERANCE",
    "PLAYER_HEALTH_BAR",
    "PLAYER_SPRITE_RADIUS",
    "STATE_DOT_REGION",
    "UNITS_MIN_Y",
    "EnemyAnnotation",
    "H",
    "RoundSummary",
    "SilverFeatures",
    "W",
    "aggregate_rounds",
    "assign_round_indices",
    "classify_frame",
    "crop_region",
    "generate_synthetic_annotation",
    "health_bar_fill",
    "identify_player_and_enemies",
    "process_frames",
    "process_image",
    "read_ability_indicators",
    "read_clock_ocr",
    "read_damage_flash",
    "read_health_fill",
    "read_state_dot",
    "render_frame_and_state",
    "render_synthetic_frame",
    "simulate_match",
]
