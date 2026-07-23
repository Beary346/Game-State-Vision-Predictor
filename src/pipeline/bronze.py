import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

H = 1440
W = 2560


@dataclass
class BronzeFeatures:
    """Metadata extracted from the raw screenshot during Bronze preprocessing.

    The image data itself is saved to disk as a PNG; only metadata and
    summary statistics are returned in this struct for downstream logging.
    """

    image_path: str
    mean_brightness: float
    std_brightness: float
    contrast: float
    width: int
    height: int
    channels: int


# ── I/O helpers ──────────────────────────────────────────────────────────────


def load_image(image_path: str) -> np.ndarray:
    """Load a raw screenshot and validate its shape.

    Returns RGB-ordered uint8 array of shape (H, W, 3).
    If the image dimensions differ from (H, W) it is resized silently
    so downstream code always receives a consistent shape.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    h, w = img.shape[:2]
    if (h, w) != (H, W):
        img = cv2.resize(img, (W, H))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save an RGB uint8 image to disk as PNG."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr)


# ── Generic OpenCV preprocessing (game-agnostic) ────────────────────────────


def denoise(img: np.ndarray, strength: int = 10) -> np.ndarray:
    """Apply Non-Local Means denoising.

    Removes sensor / compression noise while preserving edges.
    strength controls the filter strength (higher = more smoothing).
    """
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def enhance_contrast(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the
    L-channel of the LAB color space.

    Improves local contrast without amplifying noise in uniform regions.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def sharpen(img: np.ndarray, amount: float = 0.3) -> np.ndarray:
    """Apply unsharp-mask sharpening.

    amount controls the strength (0.0 = no change).
    """
    blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess(img: np.ndarray) -> np.ndarray:
    """Run the full generic preprocessing pipeline on a raw frame.

    Pipeline order:
      1. Shape validation
      2. Denoising
      3. Contrast enhancement (CLAHE)
      4. Mild sharpening

    Returns a uint8 RGB image of the same shape.
    """
    assert img.shape == (H, W, 3), f"Expected ({H}, {W}, 3), got {img.shape}"

    img = denoise(img)
    img = enhance_contrast(img)
    img = sharpen(img)

    return img


# ── Metadata extraction ──────────────────────────────────────────────────────


def extract_metadata(img: np.ndarray, image_path: str = "") -> BronzeFeatures:
    """Compute summary statistics from the preprocessed image."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return BronzeFeatures(
        image_path=image_path,
        mean_brightness=float(np.mean(gray)),
        std_brightness=float(np.std(gray)),
        contrast=float(gray.max() - gray.min()),
        width=W,
        height=H,
        channels=3,
    )


# ── End-to-end processing ────────────────────────────────────────────────────


def process_image(image_path: str, output_dir: str) -> dict:
    """Load a raw screenshot, preprocess it, and save artifacts.

    Returns a dict of paths to saved artifacts:
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

    with open(json_path, "w") as f:
        json.dump(asdict(metadata), f, indent=2)

    return {"preprocessed": png_path, "metadata": json_path}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bronze: OpenCV preprocessing")
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-o", "--output", default="data/silver", help="output directory")
    args = vars(ap.parse_args())

    result = process_image(args["image"], args["output"])
    print(f"Preprocessed image : {result['preprocessed']}")
    print(f"Metadata           : {result['metadata']}")
    print("Bronze prepreprocessing complete.")
