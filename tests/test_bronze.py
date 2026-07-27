import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.pipeline.bronze import (
    BronzeFeatures,
    H,
    W,
    denoise,
    enhance_contrast,
    extract_metadata,
    load_image,
    preprocess,
    process_image,
    save_image,
    sharpen,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def real_frame(test_screenshot_path):
    """Load the real test screenshot once per session."""
    return load_image(test_screenshot_path)


@pytest.fixture
def blank_frame():
    return np.zeros((H, W, 3), dtype=np.uint8)


# ── load_image ───────────────────────────────────────────────────────────────


def test_load_image_from_disk(test_screenshot_path):
    """Bronze must load a real screenshot successfully."""
    img = load_image(test_screenshot_path)
    assert img.shape == (H, W, 3)
    assert img.dtype == np.uint8


def test_load_image_raises_on_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(str(tmp_path / "nonexistent.png"))


def test_load_image_resizes_bad_shape(tmp_path):
    """load_image resizes mismatched dimensions to (H, W) instead of failing."""
    bad = np.zeros((100, 100, 3), dtype=np.uint8)
    bad_path = str(tmp_path / "bad.png")
    cv2.imwrite(bad_path, bad)
    img = load_image(bad_path)
    assert img.shape == (H, W, 3)


# ── preprocess pipeline ──────────────────────────────────────────────────────


class TestPreprocessPipeline:
    """All preprocessing functions must preserve shape on real data."""

    def test_denoise_preserves_shape(self, real_frame):
        out = denoise(real_frame)
        assert out.shape == (H, W, 3)
        assert out.dtype == np.uint8

    def test_enhance_contrast_preserves_shape(self, real_frame):
        out = enhance_contrast(real_frame)
        assert out.shape == (H, W, 3)
        assert out.dtype == np.uint8

    def test_sharpen_preserves_shape(self, real_frame):
        out = sharpen(real_frame)
        assert out.shape == (H, W, 3)
        assert out.dtype == np.uint8

    def test_preprocess_full_pipeline(self, real_frame):
        out = preprocess(real_frame)
        assert out.shape == (H, W, 3)
        assert out.dtype == np.uint8

    def test_preprocess_actually_changes_image(self, real_frame):
        """Preprocessing must modify the image (not a no-op)."""
        out = preprocess(real_frame)
        diff = np.abs(real_frame.astype(np.int16) - out.astype(np.int16))
        assert diff.mean() > 0.5, "Preprocessing appears to be a no-op"

    def test_preprocess_rejects_bad_shape(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(AssertionError):
            preprocess(img)


# ── extract_metadata ─────────────────────────────────────────────────────────


class TestMetadata:
    def test_extract_metadata_type(self, real_frame):
        meta = extract_metadata(real_frame)
        assert isinstance(meta, BronzeFeatures)
        assert isinstance(meta.mean_brightness, float)
        assert isinstance(meta.std_brightness, float)
        assert isinstance(meta.contrast, float)
        assert meta.width == W
        assert meta.height == H
        assert meta.channels == 3

    def test_brightness_in_expected_range(self, real_frame):
        meta = extract_metadata(real_frame)
        assert 0 <= meta.mean_brightness <= 255
        assert 0 <= meta.contrast <= 255

    def test_metadata_on_blank_frame(self, blank_frame):
        meta = extract_metadata(blank_frame)
        assert meta.mean_brightness == 0.0
        assert meta.std_brightness == 0.0
        assert meta.contrast == 0.0


# ── process_image (end-to-end) ──────────────────────────────────────────────


class TestProcessImage:
    def test_process_image_saves_artifacts(self, tmp_path, test_screenshot_path):
        result = process_image(test_screenshot_path, str(tmp_path))
        assert Path(result["preprocessed"]).exists()
        assert Path(result["metadata"]).exists()

    def test_metadata_json_is_valid(self, tmp_path, test_screenshot_path):
        result = process_image(test_screenshot_path, str(tmp_path))
        with open(result["metadata"]) as f:
            data = json.load(f)
        assert "mean_brightness" in data
        assert "std_brightness" in data
        assert "contrast" in data
        assert "width" in data

    def test_preprocessed_image_is_valid(self, tmp_path, test_screenshot_path):
        result = process_image(test_screenshot_path, str(tmp_path))
        img = cv2.imread(result["preprocessed"])
        assert img is not None
        assert img.shape == (H, W, 3)

    def test_process_image_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            process_image(str(tmp_path / "nonexistent.png"), str(tmp_path / "out"))


# ── Round-trip save/load ────────────────────────────────────────────────────


def test_save_then_load_round_trip(tmp_path, real_frame):
    out = str(tmp_path / "saved.png")
    save_image(real_frame, out)
    reloaded = load_image(out)
    np.testing.assert_array_equal(real_frame, reloaded)
