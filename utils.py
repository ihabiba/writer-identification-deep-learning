import json
import os
from typing import Dict, Tuple

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def is_png(filename: str) -> bool:
    """Return True if filename looks like a PNG image."""
    return filename.lower().endswith(".png")


def extract_writer_label(filename: str) -> str:
    """
    Extract writer label from dataset filename.
    Rule: first two characters are the writer ID (e.g., '01', '70').
    """
    return filename[:2]


def load_gray_image(path: str) -> np.ndarray:
    """
    Load image from disk as uint8 grayscale using OpenCV.
    Raises ValueError if the image cannot be loaded.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


def save_class_map(path: str, class_to_idx: Dict[str, int]) -> None:
    """
    Save class_to_idx mapping to JSON.
    Stored as: {"01": 0, "02": 1, ...}
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2, sort_keys=True)


def load_class_map(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Load class_to_idx mapping from JSON and also return idx_to_class.
    """
    with open(path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    # JSON loads keys as strings, values as ints -> ok
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class
