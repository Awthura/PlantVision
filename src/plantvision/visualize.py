"""
Overlay generation and trait chart.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def draw_overlay(image: np.ndarray, results, cfg: dict) -> np.ndarray:
    """
    Paint crop masks green and weed masks red over the original image.

    Returns a BGR uint8 array (same size as input).
    """
    crop_class: int = cfg["traits"]["crop_class"]
    weed_class: int = cfg["traits"]["weed_class"]
    crop_color = tuple(cfg["viz"]["crop_color"])
    weed_color = tuple(cfg["viz"]["weed_color"])
    alpha: float = cfg["viz"]["alpha"]

    overlay = image.copy()

    if results.masks is None:
        return overlay

    h, w = image.shape[:2]
    for i, cls in enumerate(results.boxes.cls.int().tolist()):
        color = crop_color if cls == crop_class else (weed_color if cls == weed_class else None)
        if color is None:
            continue
        mask_np = results.masks.data[i].cpu().numpy()
        mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = (mask_resized > 0.5).astype(np.uint8)
        colored = np.zeros_like(image)
        colored[:] = color
        overlay = np.where(binary[:, :, None], cv2.addWeighted(overlay, 1 - alpha, colored, alpha, 0), overlay)

    return overlay.astype(np.uint8)


def save_overlay(image_path: Path, overlay: np.ndarray, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / image_path.name
    cv2.imwrite(str(out_path), overlay)
    return out_path


def plot_canopy_chart(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of canopy cover % per image."""
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.3), 4))
    ax.bar(range(len(df)), df["canopy_cover_pct"], color="#2ecc71", edgecolor="none")
    ax.set_xlabel("Image index")
    ax.set_ylabel("Canopy cover (%)")
    ax.set_title("Per-image canopy cover — PlantVision")
    ax.set_ylim(0, 100)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    log.info("Chart saved to %s", out_path)
