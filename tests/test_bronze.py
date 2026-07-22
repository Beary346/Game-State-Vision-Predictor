import pytest
import cv2
import numpy as np
from src.pipeline.bronze import load_image, describe_image, save_image
from src.utils.image_utils import assert_valid_frame


def test_load_image_raises_on_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(str(tmp_path / "nonexistent.png"))


def test_assert_valid_frame_passes():
    img = np.zeros((1440, 2560, 3), dtype=np.uint8)
    assert_valid_frame(img)


def test_assert_valid_frame_raises_on_bad_shape():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(AssertionError):
        assert_valid_frame(img)
