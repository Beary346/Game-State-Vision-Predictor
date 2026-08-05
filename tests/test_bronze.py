import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.ingestor_bronze import (
    MAX_VOD_DURATION_SEC,
    BronzeFeatures,
    H,
    VodMetadata,
    W,
    denoise,
    enhance_contrast,
    extract_frames,
    extract_metadata,
    get_vod_info,
    load_image,
    preprocess,
    process_image,
    process_input,
    process_vod,
    save_image,
    sharpen,
    validate_vod_duration,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def real_frame(test_screenshot_path):
    """Load the real test screenshot once per session."""
    return load_image(test_screenshot_path)


@pytest.fixture
def blank_frame():
    return np.zeros((H, W, 3), dtype=np.uint8)


@pytest.fixture
def sample_vod(tmp_path):
    """Create a short synthetic VOD for testing."""
    vod_path = tmp_path / "test_vod.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    out = cv2.VideoWriter(str(vod_path), fourcc, fps, (W, H))

    for i in range(90):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            f"Frame {i}",
            (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            3,
        )
        out.write(frame)
    out.release()
    return str(vod_path)


@pytest.fixture
def long_vod(tmp_path):
    """Create a VOD that exceeds MAX_VOD_DURATION_SEC.

    Tiny 64x48 frames keep encoding cheap: the duration contract is
    (frame_count / fps) and is independent of resolution, so this fixture
    builds ~27k frames in well under a second instead of minutes.
    """
    vod_path = tmp_path / "long_vod.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    duration_sec = MAX_VOD_DURATION_SEC + 10
    total_frames = int(duration_sec * fps)
    out = cv2.VideoWriter(str(vod_path), fourcc, fps, (64, 48))

    for i in range(total_frames):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return str(vod_path)


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

    def test_metadata_includes_frame_info(self, real_frame):
        meta = extract_metadata(real_frame, frame_index=42, timestamp_sec=1.4)
        assert meta.frame_index == 42
        assert meta.timestamp_sec == 1.4


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


# ── VOD processing ───────────────────────────────────────────────────────────


class TestVodProcessing:
    def test_get_vod_info(self, sample_vod):
        info = get_vod_info(sample_vod)
        assert isinstance(info, VodMetadata)
        assert info.fps == 30.0
        assert info.total_frames == 90
        assert info.duration_sec == pytest.approx(3.0, rel=0.1)
        assert info.width == W
        assert info.height == H

    def test_validate_vod_duration_passes(self, sample_vod):
        info = get_vod_info(sample_vod)
        validate_vod_duration(info)

    def test_validate_vod_duration_fails(self, long_vod):
        info = get_vod_info(long_vod)
        with pytest.raises(ValueError, match="exceeds maximum"):
            validate_vod_duration(info)

    def test_extract_frames_default_interval(self, sample_vod, tmp_path):
        frames = extract_frames(sample_vod, str(tmp_path))
        assert len(frames) == 90
        for i, frame in enumerate(frames):
            assert frame["frame_index"] == i
            assert Path(frame["image_path"]).exists()

    def test_extract_frames_custom_interval(self, sample_vod, tmp_path):
        frames = extract_frames(sample_vod, str(tmp_path), frame_interval=30)
        assert len(frames) == 3
        assert frames[0]["frame_index"] == 0
        assert frames[1]["frame_index"] == 30
        assert frames[2]["frame_index"] == 60

    def test_extract_frames_max_frames(self, sample_vod, tmp_path):
        frames = extract_frames(sample_vod, str(tmp_path), max_frames=5)
        assert len(frames) == 5

    def test_extract_frames_resizes_to_target(self, sample_vod, tmp_path):
        frames = extract_frames(sample_vod, str(tmp_path))
        img = cv2.imread(frames[0]["image_path"])
        assert img.shape[:2] == (H, W)

    def test_process_vod_end_to_end(self, sample_vod, tmp_path):
        result = process_vod(sample_vod, str(tmp_path), frame_interval=10)
        assert "frames" in result
        assert "vod_metadata" in result
        assert len(result["frames"]) == 9
        assert result["vod_metadata"]["extracted_frames"] == 9
        assert Path(result["vod_metadata_path"]).exists()

        for frame in result["frames"]:
            assert Path(frame["preprocessed"]).exists()
            assert Path(frame["metadata"]).exists()
            with open(frame["metadata"]) as f:
                meta = json.load(f)
            assert "frame_index" in meta
            assert "timestamp_sec" in meta

    def test_process_vod_saves_vod_metadata(self, sample_vod, tmp_path):
        result = process_vod(sample_vod, str(tmp_path), frame_interval=30)
        with open(result["vod_metadata_path"]) as f:
            vod_meta = json.load(f)
        assert vod_meta["duration_sec"] == pytest.approx(3.0, rel=0.1)
        assert vod_meta["fps"] == 30.0
        assert vod_meta["total_frames"] == 90

    def test_process_vod_max_frames(self, sample_vod, tmp_path):
        result = process_vod(sample_vod, str(tmp_path), frame_interval=10, max_frames=4)
        assert len(result["frames"]) == 4
        assert result["vod_metadata"]["extracted_frames"] == 4

    def test_process_vod_single_pass_no_raw_intermediates(self, sample_vod, tmp_path):
        """Remake guarantee: no unprocessed raw frames left on disk."""
        process_vod(sample_vod, str(tmp_path), frame_interval=30)
        raw_leftovers = list(Path(tmp_path).glob("*_frame_*.png"))
        assert raw_leftovers, "expected bronze frames to exist"
        for leftover in raw_leftovers:
            assert leftover.name.endswith("_bronze.png"), (
                f"raw intermediate left behind: {leftover.name}"
            )

    def test_extract_frames_saves_raw_while_process_vod_preprocesses(self, sample_vod, tmp_path):
        """extract_frames is raw; process_vod output must differ (preprocessed)."""
        raw_dir = tmp_path / "raw"
        bronze_dir = tmp_path / "bronze"
        raw_frames = extract_frames(sample_vod, str(raw_dir), frame_interval=5)
        bronze_frames = process_vod(sample_vod, str(bronze_dir), frame_interval=5)

        raw_img = cv2.imread(raw_frames[0]["image_path"])
        bronze_img = cv2.imread(bronze_frames["frames"][0]["preprocessed"])
        diff = np.abs(raw_img.astype(np.int16) - bronze_img.astype(np.int16))
        assert diff.mean() > 0.5, "process_vod output appears identical to raw extraction"


# ── Unified process_input ────────────────────────────────────────────────────


class TestProcessInput:
    def test_process_input_image(self, test_screenshot_path, tmp_path):
        result = process_input(test_screenshot_path, str(tmp_path))
        assert "preprocessed" in result
        assert "metadata" in result

    def test_process_input_vod(self, sample_vod, tmp_path):
        result = process_input(sample_vod, str(tmp_path), frame_interval=15)
        assert "frames" in result
        assert "vod_metadata" in result
        assert len(result["frames"]) == 6

    def test_process_input_invalid_extension(self, tmp_path):
        fake = tmp_path / "file.xyz"
        fake.write_text("not an image")
        with pytest.raises(FileNotFoundError):
            process_input(str(fake), str(tmp_path))


# ── Round-trip save/load ────────────────────────────────────────────────────


def test_save_then_load_round_trip(tmp_path, real_frame):
    out = str(tmp_path / "saved.png")
    save_image(real_frame, out)
    reloaded = load_image(out)
    np.testing.assert_array_equal(real_frame, reloaded)
