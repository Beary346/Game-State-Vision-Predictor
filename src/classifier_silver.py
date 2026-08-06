"""Silver layer: HUD-region grounding per frame -> per-frame state tuples.

The Bronze layer hands us standardized (H, W, 3) frames. Silver reads the
grounded HUD regions of each frame and turns what it sees into a plain-English
per-frame state tuple: player_health, enemy_health, aggression (attacking),
defense (defending), round clock, domain readiness, damage flag, plus
OCR-confidence warnings.

Everything here is a deterministic heuristic (OpenCV + a bundled digit OCR), so
the layer can be verified on held-out synthetic frames -- the "known read
accuracy" contract from AGENTS.md -- without needing a trained model. The CNN
training path lives in src/models/silver_cnn.py and src/pipeline/silver_train.py
and is used when a model exists; heuristics are the default reader.
"""

import json
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from src.ingestor_bronze import load_image as load_frame

H = 1440
W = 2560

HEALTH_BAR_TOLERANCE = 0.06

ENEMY_HEALTH_BAR = (20, 18, 500, 30)
PLAYER_HEALTH_BAR = (W - 520, 18, 500, 30)
CLOCK_REGION = (W // 2 - 110, 14, 220, 44)
STATE_DOT_REGION = (W - 560, 62, 44, 44)
ABILITY_ROW_REGION = (W - 560, 118, 220, 44)
ABILITY_SLOTS = 4
DAMAGE_FLASH_REGION = (W // 2 - 140, 120, 280, 44)
PLAYER_SPRITE_RADIUS = 34
UNITS_MIN_Y = 170

# RGB palette for the synthetic HUD. Hue values are chosen so the HSV ranges
# used by the frame detectors land on the expected colour families.
_PLAYER_GREEN = (10, 160, 60)
_ENEMY_RED = (255, 40, 30)
_ATTACK_YELLOW = (255, 230, 20)
_DEFEND_BLUE = (60, 110, 245)
_BAR_GREEN = (0, 200, 60)
_BAR_RED = (0, 60, 220)
_FLASH_RED = (255, 30, 30)
_GRAY_OFF = (110, 110, 110)


@dataclass
class SilverFeatures:
    """Per-frame state tuple (the schema Gold and the event layer consume).

    The first eight fields are the original SilverFeatures schema and must stay
    stable: gold.py vectorises them and old JSONs deserialise. The trailing
    fields are HUD-grounding extras; all have defaults so older readers keep
    working and new readers can opt in.
    """

    image_path: str = ""
    player_health: float = 0.0
    player_position: tuple[float, float] = (0.5, 0.5)
    enemies: list = field(default_factory=list)
    attacking: bool = False
    defending: bool = False
    damage_indicator: bool = False
    num_enemies: int = 0
    # HUD-grounding extras.
    frame_index: int = 0
    timestamp_sec: float = 0.0
    clock_sec: float | None = None
    domain_ready: bool = False
    ocr_confidence: float = 1.0
    round_index: int = 0
    warnings: list = field(default_factory=list)

    def to_annotation(self) -> dict:
        """Return only the original 8-field annotation subset (CNN compat)."""
        return {
            "player_health": self.player_health,
            "player_position": list(self.player_position),
            "enemies": self.enemies,
            "attacking": self.attacking,
            "defending": self.defending,
            "damage_indicator": self.damage_indicator,
            "num_enemies": self.num_enemies,
        }


@dataclass
class RoundSummary:
    """Aggregated state over one detected round."""

    round_index: int
    start_timestamp: float
    end_timestamp: float
    num_frames: int
    min_player_health: float
    min_enemy_health: float
    avg_health_ratio: float


def crop_region(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop an (x, y, w, h) box from an RGB frame, clamped to frame bounds."""
    x, y, w, h = bbox
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return frame[y0:y1, x0:x1]


def _hsv(frame: np.ndarray) -> np.ndarray:
    """HSV view of an RGB frame (OpenCV convention, hue 0..180)."""
    return cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)


def read_health_fill(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Compute the filled fraction of a health bar plus the read confidence.

    Bar backing is dark/desaturated; the fill is saturated colour. Returns
    (fill, confidence) -- fill in [0, 1], confidence in [0, 1] describing how
    unambiguous each bar column's filled/empty verdict was.
    """
    crop = crop_region(frame, bbox)
    if crop.size == 0:
        return 0.0, 0.0
    sat = _hsv(crop)[:, :, 1].astype(np.float32)
    col_sat = sat.mean(axis=0)  # per column: the fill travels left -> right
    col_filled = col_sat > 40.0
    fill = float(col_filled.mean())
    ambiguous = float(((col_sat / 255.0 > 0.10) & (col_sat / 255.0 < 0.90)).mean())
    return fill, float(1.0 - ambiguous)


def health_bar_fill(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """Convenience wrapper: fill fraction only."""
    fill, _ = read_health_fill(frame, bbox)
    return fill


def _render_digit(d: int) -> np.ndarray:
    """Render one digit (0-9) with the same Hershey style the clock uses."""
    canvas = np.zeros((96, 128), dtype=np.uint8)
    cv2.putText(
        canvas,
        str(d),
        (14, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        255,
        2,
        cv2.LINE_AA,
    )
    ys, xs = np.where(canvas > 0)
    glyph = canvas[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    if glyph.size == 0:
        return np.zeros((40, 40), dtype=np.uint8)
    return cv2.resize(glyph, (40, 40)).astype(np.uint8)


_DIGIT_TEMPLATES = [_render_digit(d) for d in range(10)]


def _match_digit(fragment: np.ndarray) -> tuple[int | None, float]:
    """Classify a binary digit fragment against the template bank.

    Returns (digit or None, confidence). Confidence is one minus the best
    mean-absolute-difference of the binarised glyphs. Below 0.55 counts as a
    miss so garbage reads surface as None (callers warn).
    """
    if fragment.size == 0:
        return None, 0.0
    non_zero = fragment > 0
    if not non_zero.any():
        return None, 0.0
    ys, xs = np.where(non_zero)
    glyph = fragment[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    fixed = cv2.resize(glyph.astype(np.uint8), (40, 40))
    fixed_bin = (fixed > 0).astype(np.float32)

    best_digit, best_conf = None, -1.0
    for d, tmpl in enumerate(_DIGIT_TEMPLATES):
        tmpl_bin = (tmpl > 0).astype(np.float32)
        conf = 1.0 - float(np.abs(tmpl_bin - fixed_bin).mean())
        if conf > best_conf:
            best_conf, best_digit = conf, d
    return (best_digit if best_conf >= 0.55 else None, best_conf)


def read_clock_ocr(
    frame: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[float | None, float]:
    """Read a 2-digit round clock from a grounded HUD region via template OCR.

    Returns (clock_seconds in [0, 99] or None, confidence in [0, 1]). A low
    confidence read yields None; the caller surfaces the confidence as a
    warning.
    """
    crop = crop_region(frame, bbox)
    if crop.size == 0:
        return None, 0.0
    text_mask = (crop[:, :, 0].astype(np.float32) > 150).astype(np.uint8)
    _, xs = np.where(text_mask > 0)
    if xs.size == 0:
        return None, 0.0

    x_left, x_right = int(xs.min()), int(xs.max())
    col_has = text_mask[:, x_left:x_right + 1].any(axis=0)

    # Group non-empty columns into digit runs; merge runs separated by a thin gap.
    runs: list[tuple[int, int]] = []
    start = None
    for x in range(col_has.shape[0]):
        if col_has[x] and start is None:
            start = x
        elif not col_has[x] and start is not None:
            runs.append((start, x - 1))
            start = None
    if start is not None:
        runs.append((start, col_has.shape[0] - 1))

    merged: list[tuple[int, int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= 1:
            ms, _ = merged[-1]
            merged[-1] = (ms, e)
        else:
            merged.append((s, e))
    if len(merged) != 2:
        return None, 0.0  # not a clean two-digit read; caller warns

    digits: list[int] = []
    confs: list[float] = []
    for s, e in merged:
        digit, conf = _match_digit(text_mask[:, x_left + s:x_left + e + 1])
        if digit is None:
            return None, conf
        digits.append(digit)
        confs.append(conf)

    value = digits[0] * 10 + digits[1]
    if value > 99:
        return None, min(confs)
    return float(value), min(confs)


def read_ability_indicators(
    frame: np.ndarray, bbox: tuple[int, int, int, int], n_slots: int = ABILITY_SLOTS
) -> tuple[list[bool], float]:
    """Read per-slot readiness over a grounded ability/domain row.

    Returns (ready_flags, confidence). A slot counts as ready when a decent
    fraction of it is saturated colour (lit), grey/off otherwise.
    """
    crop = crop_region(frame, bbox)
    if crop.size == 0:
        return [False] * n_slots, 0.0
    slot_width = max(crop.shape[1] // n_slots, 1)
    ready: list[bool] = []
    confs: list[float] = []
    for s in range(n_slots):
        slot = crop[:, s * slot_width : (s + 1) * slot_width]
        if slot.size == 0:
            frac = 0.0
        else:
            frac = float((_hsv(slot)[:, :, 1].astype(np.float32) > 60.0).mean())
        ready.append(frac > 0.25)
        confs.append(float(1.0 - abs(frac - 0.5) / 0.5))
    return ready, float(np.mean(confs) if confs else 0.0)


def read_state_dot(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[bool, bool]:
    """Read the grounded HUD state dot: yellow -> attacking, blue -> defending."""
    dot = crop_region(frame, bbox)
    if dot.size == 0:
        return False, False
    hsv = _hsv(dot)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    lit = (sat > 90) & (val > 60)
    if not lit.any():
        return False, False
    hues = hue[lit].astype(np.float32)
    yellow = float(((hues >= 18) & (hues <= 40)).mean())
    blue = float(((hues >= 95) & (hues <= 130)).mean())
    if yellow > 0.4:
        return True, False
    if blue > 0.4:
        return False, True
    return False, False


def read_damage_flash(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    """True when a grounded damage-flash region is dominated by saturated red."""
    zone = crop_region(frame, bbox)
    if zone.size == 0:
        return False
    r = zone[:, :, 0].astype(np.float32)
    g = zone[:, :, 1].astype(np.float32)
    b = zone[:, :, 2].astype(np.float32)
    red_wins = (r - g > 60) & (r - b > 60) & (r > 120)
    return bool(red_wins.mean() > 0.12)


def _blob_centers(mask: np.ndarray, min_area: int = 300) -> list[tuple[int, int]]:
    """Centroids of closed connected blobs in a binary mask (full-frame coords)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        centers.append((cx, cy))
    return centers


def identify_player_and_enemies(frame: np.ndarray) -> tuple[tuple[float, float], list[dict]]:
    """Find the player and enemy sprites on a frame.

    Identification rules (AGENTS.md): the player is anyone relatively centre
    screen with nothing overhead (no floating name, no overhead health bar),
    while enemies carry overhead red health bars. Green = player-ish, red =
    enemy-ish, then the player is the green blob nearest screen centre. If no
    green blob is found, the nearest red enemy stands in for the player.

    Returns (player_position in [0,1]x[0,1], list of enemy annotations).
    """
    hsv = _hsv(frame)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = ((sat > 80) & (val > 80) & (hue > 35) & (hue <= 100)).astype(np.uint8) * 255
    red = ((sat > 80) & (val > 80) & ((hue <= 10) | (hue >= 170))).astype(np.uint8) * 255
    green[:UNITS_MIN_Y, :] = 0  # HUD chrome may be coloured; ignore it here
    red[:UNITS_MIN_Y, :] = 0

    player_centers = _blob_centers(green)
    enemy_centers = _blob_centers(red)
    enemies = [
        {
            "bbox": [
                float(cx - PLAYER_SPRITE_RADIUS),
                float(cy - PLAYER_SPRITE_RADIUS),
                float(2 * PLAYER_SPRITE_RADIUS),
                float(2 * PLAYER_SPRITE_RADIUS),
            ],
            "health": health_bar_fill(frame, ENEMY_HEALTH_BAR),
            "health_bar_bbox": list(ENEMY_HEALTH_BAR),
            "confidence": 1.0,
        }
        for cx, cy in enemy_centers
    ]

    player = None
    if player_centers:
        player = min(player_centers, key=lambda p: (p[0] - W / 2.0) ** 2 + (p[1] - H / 2.0) ** 2)
    elif enemies:
        player = min(enemy_centers, key=lambda p: (p[0] - W / 2.0) ** 2 + (p[1] - H / 2.0) ** 2)

    if player is not None:
        px, py = player
        # A green blob that overlaps an enemy bbox is the same character; don't
        # double count it. Fallback case: drop the red blob we promoted.
        enemies = [
            e
            for e in enemies
            if not (
                e["bbox"][0] <= px <= e["bbox"][0] + e["bbox"][2]
                and e["bbox"][1] <= py <= e["bbox"][1] + e["bbox"][3]
            )
        ]

    player_position = (float(px / W), float(py / H)) if player else (0.5, 0.5)
    return player_position, enemies


def classify_frame(
    frame: np.ndarray,
    image_path: str = "",
    frame_index: int = 0,
    timestamp_sec: float = 0.0,
    warn_threshold: float = 0.55,
) -> SilverFeatures:
    """Read one (H, W, 3) frame and emit a SilverFeatures state tuple."""
    frame = np.asarray(frame)
    assert frame.shape == (H, W, 3), f"expected (H, W, 3) frame, got {frame.shape}"

    warnings: list[str] = []
    player_health, health_conf = read_health_fill(frame, PLAYER_HEALTH_BAR)
    player_position, enemies = identify_player_and_enemies(frame)
    attacking, defending = read_state_dot(frame, STATE_DOT_REGION)
    damage_flash = read_damage_flash(frame, DAMAGE_FLASH_REGION)
    ready, ability_conf = read_ability_indicators(frame, ABILITY_ROW_REGION)
    domain_ready = bool(all(ready))

    # There is no round clock in regular game modes (Jujutsu Shenanigans
    # duels have none), so the clock region is not a Silver feature: we never
    # OCR it, never warn about it, and exclude it from OCR confidence.
    clock_sec = None

    confs = [health_conf, ability_conf]
    min_conf = float(np.min(confs))
    if min_conf < warn_threshold:
        warnings.append(f"low OCR confidence {min_conf:.2f}")

    return SilverFeatures(
        image_path=image_path,
        player_health=player_health,
        player_position=player_position,
        enemies=enemies,
        attacking=attacking,
        defending=defending,
        damage_indicator=damage_flash,
        num_enemies=len(enemies),
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        clock_sec=clock_sec,
        domain_ready=domain_ready,
        ocr_confidence=min_conf,
        round_index=0,
        warnings=warnings,
    )


def _is_new_round(state: SilverFeatures, prev_health: float | None) -> bool:
    """Detect a round boundary from the state deltas: a big health heal-up.

    The recorded game modes have no round clock, so a new round is signalled
    only by the health bar snapping back up (respawn / round reset). A heal of
    more than ``+0.55`` between consecutive frames means the previous round
    ended (the player died, or a round was reset).
    """
    return bool(prev_health is not None and state.player_health > prev_health + 0.55)


def assign_round_indices(states: list[SilverFeatures]) -> list[SilverFeatures]:
    """Tag each state tuple with a monotonic round index (0-based)."""
    tagged: list[SilverFeatures] = []
    current = 0
    prev_health: float | None = None
    for state in states:
        if tagged and _is_new_round(state, prev_health):
            current += 1
        tagged.append(replace(state, round_index=current))
        prev_health = state.player_health
    return tagged


def _round_summary(index: int, frames: list[SilverFeatures]) -> RoundSummary:
    enemy_means = [
        float(np.mean([e["health"] for e in f.enemies])) if f.enemies else 0.0 for f in frames
    ]
    return RoundSummary(
        round_index=index,
        start_timestamp=min(f.timestamp_sec for f in frames),
        end_timestamp=max(f.timestamp_sec for f in frames),
        num_frames=len(frames),
        min_player_health=min(f.player_health for f in frames),
        min_enemy_health=float(np.min(enemy_means)) if enemy_means else 0.0,
        avg_health_ratio=float(
            np.mean(
                [f.player_health / enemy if enemy else 1.0 for f, enemy in zip(frames, enemy_means)]
            )
        ),
    )


def aggregate_rounds(
    states: list[SilverFeatures],
) -> tuple[list[SilverFeatures], list[RoundSummary]]:
    """Split a frame stream into round-bundled summaries."""
    tagged = assign_round_indices(states)
    groups: dict[int, list[SilverFeatures]] = {}
    for state in tagged:
        groups.setdefault(state.round_index, []).append(state)
    summaries = [_round_summary(idx, frames) for idx, frames in sorted(groups.items())]
    return tagged, summaries


def _write_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _read_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def process_image(
    frame: np.ndarray, image_path: str = "", timestamp_sec: float = 0.0
) -> SilverFeatures:
    """Single-frame state check: one screenshot in, one state tuple out."""
    return classify_frame(frame, image_path=image_path, timestamp_sec=timestamp_sec)


def process_frames(bronze_dir: Path | str, output_dir: Path | str) -> dict:
    """Run Silver over a Bronze frame stream and write per-frame JSON + rounds.

    Expects Bronze-contract frame files (``*_bronze.png``, e.g.
    ``<stem>_frame_%06d_bronze.png`` from ``process_vod``) with matching
    sidecar JSONs (the Bronze contract). Writes ``*_silver.json`` for every
    frame plus a ``rounds.json`` aggregation into ``output_dir``. Damage also
    fires when the player's health drops sharply between consecutive frames.
    """
    bronze_dir = Path(bronze_dir)
    output_dir = Path(output_dir)
    pngs = sorted(bronze_dir.glob("*_bronze.png"))
    if not pngs:
        raise ValueError(f"no bronze frames (*_bronze.png) found in {bronze_dir}")

    states: list[SilverFeatures] = []
    prev_health: float | None = None
    for png in pngs:
        meta = _read_json(png.with_suffix(".json"))
        frame_index = int(meta.get("frame_index", 0))
        timestamp_sec = float(meta.get("timestamp_sec", 0.0))
        frame = load_frame(str(png))
        state = classify_frame(
            frame, image_path=str(png), frame_index=frame_index, timestamp_sec=timestamp_sec
        )
        if prev_health is not None and state.player_health < prev_health - 0.06:
            state = replace(state, damage_indicator=True)
        prev_health = state.player_health
        states.append(state)

    tagged, summaries = aggregate_rounds(states)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_artifacts = []
    for state in tagged:
        name = Path(state.image_path).name.replace("_bronze.png", "_silver.json")
        json_path = output_dir / name
        _write_json(json_path, asdict(state))
        frame_artifacts.append(
            {
                "frame_index": state.frame_index,
                "timestamp_sec": state.timestamp_sec,
                "round_index": state.round_index,
                "features_json": str(json_path),
            }
        )

    rounds_path = output_dir / "rounds.json"
    _write_json(rounds_path, [asdict(r) for r in summaries])
    return {
        "frames": frame_artifacts,
        "rounds": [asdict(r) for r in summaries],
        "rounds_json": str(rounds_path),
    }


def _new_frame() -> np.ndarray:
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:, :] = (40, 44, 52)
    return frame


def _draw_health_bar(
    img: np.ndarray, bbox: tuple[int, int, int, int], fill: float, low_health: bool = False
) -> None:
    x, y, w, h = bbox
    bar_fill = max(0.0, min(1.0, fill))
    cv2.rectangle(img, (x, y), (x + w, y + h), (12, 12, 14), -1)
    fill_color = _BAR_RED if low_health else _BAR_GREEN
    cv2.rectangle(img, (x, y), (x + int(w * bar_fill), y + h), fill_color, -1)


def _draw_clock(img: np.ndarray, seconds: float) -> None:
    text = f"{int(seconds):02d}"
    org = (W // 2 - 40, 46)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.6, (235, 235, 235), 2, cv2.LINE_AA)


def _draw_state_dot(img: np.ndarray, attacking: bool, defending: bool) -> None:
    x, y, w, h = STATE_DOT_REGION
    cx, cy = x + w // 2, y + h // 2
    color = _ATTACK_YELLOW if attacking else (_DEFEND_BLUE if defending else _PLAYER_GREEN)
    cv2.circle(img, (cx, cy), 14, color, -1)


def _draw_ability_row(img: np.ndarray, ready: list[bool]) -> None:
    x, y, w, h = ABILITY_ROW_REGION
    n = max(len(ready), 1)
    slot_w = w // n
    for i, on in enumerate(ready):
        sx = x + i * slot_w + 4
        color = _ATTACK_YELLOW if on else _GRAY_OFF
        cv2.rectangle(img, (sx, y), (sx + slot_w - 8, y + h), color, -1)


def _draw_damage_flash(img: np.ndarray, active: bool) -> None:
    if not active:
        return
    x, y, w, h = DAMAGE_FLASH_REGION
    cv2.rectangle(img, (x, y), (x + w, y + h), _FLASH_RED, -1)


def render_frame_and_state(
    player_health: float = 1.0,
    enemy_healths: list[float] | None = None,
    attacking: bool = False,
    defending: bool = False,
    damaged: bool = False,
    domain_ready: bool | None = None,
    ability_ready: list[bool] | None = None,
    clock_sec: float = 60.0,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Deterministically render a synthetic HUD frame + its ground-truth state.

    The returned (frame, state) pair feeds both the classifier tests (does the
    heuristic read match the ground truth?) and the CNN training path. With a
    fixed seed every sprite position and flag is reproducible.
    """
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    if enemy_healths is None:
        enemy_healths = [round(rng.uniform(0.0, 1.0), 2) for _ in range(rng.randint(1, 3))]

    if domain_ready is None:
        ability_ready = [rng.random() < 0.55 for _ in range(ABILITY_SLOTS)]
        domain_ready = bool(all(ability_ready))
    elif ability_ready is None:
        ability_ready = [bool(domain_ready)] * ABILITY_SLOTS

    frame = _new_frame()
    _draw_health_bar(frame, PLAYER_HEALTH_BAR, player_health, low_health=player_health < 0.3)
    _draw_health_bar(frame, ENEMY_HEALTH_BAR, float(np.mean(enemy_healths)))
    _draw_clock(frame, clock_sec)
    _draw_state_dot(frame, attacking, defending)
    _draw_ability_row(frame, ability_ready)
    _draw_damage_flash(frame, damaged)

    r = PLAYER_SPRITE_RADIUS
    px, py = W // 2, H - 170
    cv2.circle(frame, (px, py), r, _PLAYER_GREEN, -1)
    enemy_centers = [
        (int(nrng.uniform(0.3, 0.7) * W), int(nrng.uniform(0.3, 0.5) * H)) for _ in enemy_healths
    ]
    for ex, ey in enemy_centers:
        cv2.circle(frame, (ex, ey), r, _ENEMY_RED, -1)

    state = {
        "player_health": player_health,
        "player_position": [float(px / W), float(py / H)],
        "enemies": [
            {
                "bbox": [float(ex - r), float(ey - r), float(2 * r), float(2 * r)],
                "health": eh,
                "health_bar_bbox": [ex - r, ey - r - 14, 2 * r, 6],
                "confidence": 1.0,
            }
            for (ex, ey), eh in zip(enemy_centers, enemy_healths)
        ],
        "attacking": attacking,
        "defending": defending,
        "damage_indicator": damaged,
        "num_enemies": len(enemy_healths),
        "domain_ready": domain_ready,
        "clock_sec": clock_sec,
    }
    return frame, state


def render_synthetic_frame(num_enemies: int | None = None, seed: int | None = None) -> np.ndarray:
    """Render a synthetic frame (legacy signature, silver_train/EDA compatible)."""
    rng = random.Random(seed)
    if num_enemies is None:
        num_enemies = rng.randint(1, 3)
    enemy_healths = [round(rng.uniform(0.2, 1.0), 2) for _ in range(num_enemies)]
    return render_frame_and_state(
        player_health=round(rng.uniform(0.3, 1.0), 2),
        enemy_healths=enemy_healths,
        attacking=rng.random() < 0.5,
        defending=rng.random() < 0.5,
        damaged=rng.random() < 0.2,
        clock_sec=float(rng.randint(5, 99)),
        seed=seed,
    )[0]


def generate_synthetic_annotation(num_enemies: int | None = None, seed: int = 0) -> dict:
    """Deterministic annotation dict for one synthetic frame (8-field schema).

    The ``seed`` must match the one handed to ``render_synthetic_frame`` so the
    rendered frame and the annotation agree on the enemy count/positions.
    """
    rng = random.Random(seed)
    if num_enemies is None:
        num_enemies = rng.randint(1, 3)
    r = PLAYER_SPRITE_RADIUS
    enemy_centers = [
        (int(rng.uniform(0.3, 0.7) * W), int(rng.uniform(0.3, 0.5) * H)) for _ in range(num_enemies)
    ]
    return {
        "player_health": round(rng.uniform(0.3, 1.0), 2),
        "player_position": [0.5, (H - 170) / H],
        "enemies": [
            {
                "bbox": [float(cx - r), float(cy - r), float(2 * r), float(2 * r)],
                "health": round(rng.uniform(0.2, 1.0), 2),
                "health_bar_bbox": [cx - r, cy - r - 14, 2 * r, 6],
                "confidence": 1.0,
            }
            for cx, cy in enemy_centers
        ],
        "attacking": bool(rng.random() < 0.5),
        "defending": bool(rng.random() < 0.5),
        "damage_indicator": bool(rng.random() < 0.3),
        "num_enemies": num_enemies,
    }


def simulate_match(num_frames: int = 40, fps: int = 30, seed: int = 0) -> list[dict]:
    """A short deterministic match: list of {frame, state, meta} entries.

    Two enemies at 0.75/0.6 health, round resets every 15 frames (clock rewinds
    to 60), attacking/defending toggles on fixed beats. Used by tests and by
    notebook 02 to walk the full Bronze -> Silver flow without real footage.
    """
    rng = random.Random(seed)
    clock = 60.0
    player_health = 1.0
    entries = []
    for i in range(num_frames):
        if i % 15 == 0:
            clock = 60.0
            player_health = 1.0
        clock = max(0.0, clock - 1.0)
        player_health = max(0.0, player_health - 0.02 * rng.random())

        frame, state = render_frame_and_state(
            player_health=player_health,
            enemy_healths=[0.75, 0.6],
            attacking=(i % 7 == 0) or (i % 9 == 4),
            defending=i % 11 == 6,
            damaged=rng.random() < 0.12,
            domain_ready=i % 13 == 9,
            clock_sec=clock,
            seed=seed + i,
        )
        entries.append(
            {
                "frame": frame,
                "state": state,
                "frame_index": i,
                "timestamp_sec": i / fps,
                "clock_sec": clock,
            }
        )
    return entries


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
