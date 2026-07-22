import argparse
from pathlib import Path

import cv2

H = 1440
W = 2560


def load_image(image_path: str) -> cv2.Mat:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    assert img.shape == (H, W, 3), f"Expected ({H}, {W}, 3), got {img.shape}"
    return img


def describe_image(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    h, w, c = img.shape[:3]
    return {"width": w, "height": h, "channels": c}


def save_image(image: cv2.Mat, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-o", "--output", default="newimage.jpg", help="output path")
    args = vars(ap.parse_args())

    img = load_image(args["image"])
    desc = describe_image(args["image"])
    print(f"width: {desc['width']} pixels")
    print(f"height: {desc['height']} pixels")
    print(f"channels: {desc['channels']}")

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    save_image(img, args["output"])
