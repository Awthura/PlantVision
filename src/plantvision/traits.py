"""
Trait extraction from YOLO segmentation masks.

Must-have : canopy cover % = crop-mask pixels / total image pixels
Bonus      : plant/stand count via crop instance count
"""

from __future__ import annotations

import numpy as np


def canopy_cover(mask: np.ndarray, crop_class: int) -> float:
    """
    Return canopy cover as a fraction in [0, 1].

    Parameters
    ----------
    mask : H×W uint8 array where each pixel = class index (0=bg, 1=crop, 2=weed)
    crop_class : class index that represents crop
    """
    if mask.size == 0:
        return 0.0
    return float(np.sum(mask == crop_class) / mask.size)


def canopy_cover_from_results(results, crop_class: int, image_shape: tuple[int, int]) -> float:
    """
    Compute canopy cover from an ultralytics Results object.

    Combines all crop-class instance masks into one binary map, then
    divides by image area.
    """
    h, w = image_shape
    total_pixels = h * w

    if results.masks is None:
        return 0.0

    crop_pixels = 0
    for i, cls in enumerate(results.boxes.cls.int().tolist()):
        if cls == crop_class:
            # masks.data is (N, H, W) float32 in [0,1]
            binary = (results.masks.data[i].cpu().numpy() > 0.5).astype(np.uint8)
            # masks may be at model resolution; resize is handled by ultralytics
            crop_pixels += int(binary.sum())

    return crop_pixels / total_pixels if total_pixels > 0 else 0.0


def plant_count(results, crop_class: int) -> int:
    """Count distinct crop instances (stand count proxy)."""
    if results.boxes is None:
        return 0
    return int(sum(1 for cls in results.boxes.cls.int().tolist() if cls == crop_class))
