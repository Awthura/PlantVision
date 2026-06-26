"""
Unit tests for trait math — uses synthetic masks only (no model required).
"""

import numpy as np
import pytest

from plantvision.traits import canopy_cover, plant_count


# ── canopy_cover ────────────────────────────────────────────────────────────

def test_canopy_cover_all_crop():
    mask = np.ones((100, 100), dtype=np.uint8)  # class 1 = crop
    assert canopy_cover(mask, crop_class=1) == pytest.approx(1.0)


def test_canopy_cover_none():
    mask = np.zeros((100, 100), dtype=np.uint8)  # all background
    assert canopy_cover(mask, crop_class=1) == pytest.approx(0.0)


def test_canopy_cover_half():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:50, :] = 1  # top half = crop
    assert canopy_cover(mask, crop_class=1) == pytest.approx(0.5)


def test_canopy_cover_quarter():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:25, :25] = 1  # 625 / 10000
    assert canopy_cover(mask, crop_class=1) == pytest.approx(0.0625)


def test_canopy_cover_ignores_weed():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, :] = 2   # weed row
    mask[1, :] = 1   # crop row
    # 10 crop pixels out of 100
    assert canopy_cover(mask, crop_class=1) == pytest.approx(0.10)


def test_canopy_cover_empty_mask():
    mask = np.zeros((0, 0), dtype=np.uint8)
    assert canopy_cover(mask, crop_class=1) == pytest.approx(0.0)


def test_canopy_cover_single_pixel():
    mask = np.array([[1]], dtype=np.uint8)
    assert canopy_cover(mask, crop_class=1) == pytest.approx(1.0)


# ── plant_count via mock Results ────────────────────────────────────────────

class _MockBoxes:
    def __init__(self, classes: list[int]):
        import torch
        self.cls = torch.tensor(classes, dtype=torch.float32)


class _MockResults:
    def __init__(self, classes: list[int]):
        self.boxes = _MockBoxes(classes)
        self.masks = None


def test_plant_count_none():
    r = _MockResults([])
    assert plant_count(r, crop_class=0) == 0


def test_plant_count_all_crop():
    r = _MockResults([0, 0, 0])
    assert plant_count(r, crop_class=0) == 3


def test_plant_count_mixed():
    r = _MockResults([0, 1, 0, 2, 0])  # 3 crop, 1 weed, 1 other
    assert plant_count(r, crop_class=0) == 3


def test_plant_count_no_crop():
    r = _MockResults([1, 1, 2])
    assert plant_count(r, crop_class=0) == 0
