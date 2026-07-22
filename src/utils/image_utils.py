import cv2

H = 1440
W = 2560


def assert_valid_frame(img: cv2.Mat) -> None:
    assert img.shape == (H, W, 3), f"Expected ({H}, {W}, 3), got {img.shape}"


def to_grayscale(img: cv2.Mat) -> cv2.Mat:
    assert_valid_frame(img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize(img: cv2.Mat, scale: float) -> cv2.Mat:
    assert_valid_frame(img)
    return cv2.resize(img, None, fx=scale, fy=scale)
