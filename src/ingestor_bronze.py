"""Bronze layer: frame extraction and OpenCV preprocessing.

Handles both single-frame screenshots and multi-minute VODs uniformly so the
Silver layer always receives the same shape contract ``(H, W, 3)``.

- ``process_image`` : one raw screenshot  -> one preprocessed PNG + metadata JSON
- ``process_vod``   : a match VOD         -> a bronze frame PNG + metadata JSON
  per sampled frame, plus a VOD-level metadata JSON describing the source file.
- ``process_input`` : routes an arbitrary path to the right processor by extension.

All extraction is single-pass: a sampled frame is converted, resized, denoised /
enhanced / sharpened and written exactly once. Nothing is staged as an
intermediate raw frame, so multi-minute VODs stay cheap on both I/O and disk.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

# ── Shape contract ───────────────────────────────────────────────────────────
H = 1440
W = 2560

# A ~15 minute VOD at desktop playback is the practical ceiling: anything longer
# fails gracefully instead of melting down the disk or the pipeline.
MAX_VOD_DURATION_SEC = 15 * 60

DEFAULT_FRAME_INTERVAL = 1

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}


@dataclass
class BronzeFeatures:
    """Per-frame metadata summary computed from the preprocessed image.

    The actual pixel data is written to disk as a PNG; this struct carries the
    summary statistics that let downstream layers reason about a frame without
    holding the full array in memory.
    """

    image_path: str
    frame_index: int
    timestamp_sec: float
    mean_brightness: float
    std_brightness: float
    contrast: float
    width: int
    height: int
    channels: int


@dataclass
class VodMetadata:
    """Metadata describing the source VOD and how much of it was sampled."""

    vod_path: str
    duration_sec: float
    fps: float
    total_frames: int
    width: int
    height: int
    frame_interval: int
    extracted_frames: int


# ── Image I/O ────────────────────────────────────────────────────────────────


def load_image(image_path: str) -> np.ndarray:
    """Load a raw screenshot and conform it to the (H, W, 3) shape contract.

    Returns an RGB-ordered uint8 array. If the source dimensions differ from
    (H, W) they are silently resized so downstream code never sees a surprise.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    return _frame_to_standard_rgb(img)


def save_image(image: np.ndarray, output_path: str) -> None:
    """Write an RGB uint8 array to disk as PNG, creating parent dirs first."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr)


def _frame_to_standard_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR frame/screenshot to the contracted RGB (H, W, 3) shape."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if (h, w) != (H, W):
        rgb = cv2.resize(rgb, (W, H))
    return rgb.astype(np.uint8)


# ── Generic OpenCV preprocessing (game-agnostic) ────────────────────────────


def denoise(img: np.ndarray, strength: int = 10) -> np.ndarray:
    """Apply Non-Local Means denoising.

    Removes sensor / compression noise while preserving edges. ``strength``
    controls how aggressively the filter smooths (higher = softer image).
    """
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def enhance_contrast(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the
    L-channel of the LAB color space.

    Boosts local contrast without amplifying noise in flat regions such as
    health bars or ability icons.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def sharpen(img: np.ndarray, amount: float = 0.3) -> np.ndarray:
    """Apply unsharp-mask sharpening to restore edges softened by denoising.

    ``amount`` controls the strength (0.0 = no change).
    """
    blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess(img: np.ndarray) -> np.ndarray:
    """Run the full generic preprocessing pipeline on a raw frame.

    Order:
      1. Shape validation (asserts the (H, W, 3) contract)
      2. Denoising (Non-Local Means)
      3. Contrast enhancement (CLAHE on L-channel)
      4. Mild sharpening (unsharp mask)

    Returns a uint8 RGB image of the same shape.
    """
    assert img.shape == (H, W, 3), f"Expected ({H}, {W}, 3), got {img.shape}"

    img = denoise(img)
    img = enhance_contrast(img)
    img = sharpen(img)

    return img


def extract_metadata(
    img: np.ndarray,
    image_path: str = "",
    frame_index: int = 0,
    timestamp_sec: float = 0.0,
) -> BronzeFeatures:
    """Compute summary statistics from a preprocessed frame."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return BronzeFeatures(
        image_path=image_path,
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        mean_brightness=float(np.mean(gray)),
        std_brightness=float(np.std(gray)),
        contrast=float(gray.max() - gray.min()),
        width=W,
        height=H,
        channels=3,
    )


# ── VOD metadata & validation ────────────────────────────────────────────────


def get_vod_info(vod_path: str) -> VodMetadata:
    """Extract metadata from a VOD file without touching its frames."""
    cap = cv2.VideoCapture(vod_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open VOD at {vod_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    cap.release()

    return VodMetadata(
        vod_path=str(Path(vod_path).resolve()),
        duration_sec=duration_sec,
        fps=fps,
        total_frames=total_frames,
        width=width,
        height=height,
        frame_interval=DEFAULT_FRAME_INTERVAL,
        extracted_frames=0,
    )


def validate_vod_duration(vod_metadata: VodMetadata) -> None:
    """Raise if a VOD exceeds the maximum allowed duration."""
    if vod_metadata.duration_sec > MAX_VOD_DURATION_SEC:
        raise ValueError(
            f"VOD duration {vod_metadata.duration_sec:.1f}s exceeds maximum "
            f"of {MAX_VOD_DURATION_SEC}s ({MAX_VOD_DURATION_SEC // 60:.0f} minutes)"
        )


def _sampled_frames(
    vod_path: str, frame_interval: int, max_frames: int | None
) -> list[tuple[int, float, np.ndarray]]:
    """Single forward pass over a VOD returning only sampled, standardized frames.

    Each sampled frame is returned as a ``(frame_index, timestamp_sec, rgb)``
    tuple where ``rgb`` is already standardized to the (H, W, 3) contract.
    Frames between samples are skipped without being materialized, keeping
    multi-minute VODs cheap. The duration limit is enforced up front so an
    over-long VOD fails before any pixels are written to disk.
    """
    vod_meta = get_vod_info(vod_path)
    validate_vod_duration(vod_meta)
    fps = vod_meta.fps

    cap = cv2.VideoCapture(vod_path)
    try:
        frames: list[tuple[int, float, np.ndarray]] = []
        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps if fps > 0 else 0.0
                frames.append((frame_idx, timestamp, _frame_to_standard_rgb(frame_bgr)))
                if max_frames and len(frames) >= max_frames:
                    break

            frame_idx += 1
    finally:
        cap.release()

    return frames


# ── Single screenshot processing ─────────────────────────────────────────────


def process_image(image_path: str, output_dir: str) -> dict:
    """Load a raw screenshot, preprocess it, and save its artifacts.

    Returns a dict of saved artifact paths:

      - ``preprocessed``: the cleaned PNG image
      - ``metadata``:     JSON file with summary statistics
    """
    img = load_image(image_path)
    cleaned = preprocess(img)
    metadata = extract_metadata(cleaned, image_path=str(Path(image_path).resolve()))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    png_path = str(out_dir / f"{stem}_bronze.png")
    json_path = str(out_dir / f"{stem}_bronze.json")

    save_image(cleaned, png_path)
    _write_json(json_path, asdict(metadata))

    return {"preprocessed": png_path, "metadata": json_path}


# ── VOD processing ───────────────────────────────────────────────────────────


def extract_frames(
    vod_path: str,
    output_dir: str,
    frame_interval: int = DEFAULT_FRAME_INTERVAL,
    max_frames: int | None = None,
) -> list[dict]:
    """Extract raw (unprocessed) frames from a VOD at a regular interval.

    ``raw`` frames carry only the standardized RGB frame and its timeline
    position --- no denoising, no enhancement. Use this when you want to inspect
    source frames without paying for heavy preprocessing on frames nobody will
    label.

    Args:
        vod_path: Path to the video file.
        output_dir: Directory to save extracted frames.
        frame_interval: Extract every Nth frame (1 = every frame).
        max_frames: Optional cap on number of extracted frames.

    Returns:
        List of dicts with keys: frame_index, timestamp_sec, image_path.
    """
    sampled = _sampled_frames(vod_path, frame_interval, max_frames)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(vod_path).stem
    extracted = []
    for frame_idx, timestamp, frame_rgb in sampled:
        frame_filename = f"{stem}_frame_{frame_idx:06d}.png"
        frame_path = str(out_dir / frame_filename)
        save_image(frame_rgb, frame_path)

        extracted.append(
            {
                "frame_index": frame_idx,
                "timestamp_sec": timestamp,
                "image_path": frame_path,
            }
        )

    return extracted


def process_vod(
    vod_path: str,
    output_dir: str,
    frame_interval: int = DEFAULT_FRAME_INTERVAL,
    max_frames: int | None = None,
) -> dict:
    """Sample, preprocess, and persist bronze frames for a whole VOD.

    Each sampled frame goes through the exact same pipeline as a screenshot
    (preprocess -> metadata) and is written once as ``{stem}_frame_%06d_bronze.png``
    plus a sidecar metadata JSON. A VOD-level metadata JSON is also written so
    the Silver layer can re-hydrate timing and sampling context.

    Returns:
        A dict with ``frames`` (per-frame artifact list),
        ``vod_metadata`` (VodMetadata as dict) and ``vod_metadata_path``.
    """
    vod_meta = get_vod_info(vod_path)
    validate_vod_duration(vod_meta)

    sampled = _sampled_frames(vod_path, frame_interval, max_frames)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(vod_path).stem
    frame_artifacts = []
    for frame_idx, timestamp, frame_rgb in sampled:
        cleaned = preprocess(frame_rgb)
        png_path = str(out_dir / f"{stem}_frame_{frame_idx:06d}_bronze.png")
        json_path = str(out_dir / f"{stem}_frame_{frame_idx:06d}_bronze.json")

        save_image(cleaned, png_path)
        metadata = extract_metadata(
            cleaned,
            image_path=png_path,
            frame_index=frame_idx,
            timestamp_sec=timestamp,
        )
        _write_json(json_path, asdict(metadata))

        frame_artifacts.append(
            {
                "frame_index": frame_idx,
                "timestamp_sec": timestamp,
                "preprocessed": png_path,
                "metadata": json_path,
            }
        )

    vod_meta.frame_interval = frame_interval
    vod_meta.extracted_frames = len(frame_artifacts)

    vod_json_path = str(out_dir / f"{stem}_vod_metadata.json")
    _write_json(vod_json_path, asdict(vod_meta))

    return {
        "frames": frame_artifacts,
        "vod_metadata": asdict(vod_meta),
        "vod_metadata_path": vod_json_path,
    }


# ── Unified entry points ─────────────────────────────────────────────────────


def process_input(
    input_path: str,
    output_dir: str,
    frame_interval: int = DEFAULT_FRAME_INTERVAL,
    max_frames: int | None = None,
) -> dict:
    """Process either an image or a VOD, switching on file extension.

    Deterministic for the file reading. Returns the same artifact dict as the
    underlying processor it routed to.
    """
    ext = Path(input_path).suffix.lower()

    if ext in VIDEO_EXTENSIONS:
        return process_vod(input_path, output_dir, frame_interval, max_frames)

    return process_image(input_path, output_dir)


# ── Private helpers ─────────────────────────────────────────────────────────


def _write_json(path: str, payload: dict) -> None:
    """Serialize a metadata dict to a human-readable JSON file."""
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Bronze: OpenCV preprocessing for screenshots and VODs"
    )
    ap.add_argument("-i", "--input", required=True, help="path to input image or VOD")
    ap.add_argument("-o", "--output", default="data/silver", help="output directory")
    ap.add_argument(
        "--frame-interval",
        type=int,
        default=DEFAULT_FRAME_INTERVAL,
        help="extract every Nth frame from a VOD (default 1 = every frame)",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="cap on number of frames extracted from a VOD",
    )
    args = ap.parse_args()

    result = process_input(
        args.input,
        args.output,
        frame_interval=args.frame_interval,
        max_frames=args.max_frames,
    )

    if "frames" in result:
        print(f"VOD processed: {result['vod_metadata']['extracted_frames']} frames extracted")
        print(f"VOD metadata : {result['vod_metadata_path']}")
        for frame in result["frames"][:3]:
            print(
                f"  Frame {frame['frame_index']} @ {frame['timestamp_sec']:.2f}s "
                f"-> {frame['preprocessed']}"
            )
        remaining = len(result["frames"]) - 3
        if remaining > 0:
            print(f"  ... and {remaining} more frames")
    else:
        print(f"Preprocessed image : {result['preprocessed']}")
        print(f"Metadata           : {result['metadata']}")

    print("Bronze preprocessing complete.")
