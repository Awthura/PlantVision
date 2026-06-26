"""
PhenoBench dataset ingestion — verifies structure and emits basic stats.

PhenoBench layout expected under data.root:
  train/images/*.png
  train/semantics/*.png   (uint8, 0=bg, 1=crop, 2=weed per class map)
  val/images/*.png
  val/semantics/*.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

log = logging.getLogger(__name__)


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def verify_split(split_dir: Path, split: str = "train") -> dict:
    """Return basic stats for one PhenoBench split and raise if files are missing."""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "semantics"

    images = sorted(images_dir.glob("*.png"))
    labels = sorted(labels_dir.glob("*.png"))

    if not images:
        raise FileNotFoundError(
            f"No images found in {images_dir}. "
            "Run scripts/download_data.py first."
        )

    paired = {p.stem for p in images} & {p.stem for p in labels}
    log.info(
        "[%s] %d images, %d masks, %d paired",
        split,
        len(images),
        len(labels),
        len(paired),
    )
    return {"split": split, "n_images": len(images), "n_paired": len(paired)}


def dataset_stats(cfg: dict) -> dict:
    """Quick pixel-class histogram over a random sample (max 50 images)."""
    train_dir = Path(cfg["data"]["train"])
    images = sorted((train_dir / "images").glob("*.png"))
    sample = images[:50]

    counts: dict[int, int] = {}
    for img_path in sample:
        mask_path = train_dir / "semantics" / img_path.name
        if not mask_path.exists():
            continue
        arr = np.array(Image.open(mask_path))
        for cls, cnt in zip(*np.unique(arr, return_counts=True)):
            counts[int(cls)] = counts.get(int(cls), 0) + int(cnt)

    total = sum(counts.values()) or 1
    stats = {k: round(v / total * 100, 2) for k, v in sorted(counts.items())}
    log.info("Pixel class distribution (%%): %s", stats)
    return stats


def build_yolo_dataset_yaml(cfg: dict, out_path: str | Path = "configs/phenobench_yolo.yaml") -> Path:
    """
    Write a YOLO-compatible dataset YAML pointing to PhenoBench splits.
    YOLO expects masks in the same relative layout; this file is the entrypoint.
    """
    out_path = Path(out_path)
    data_root = Path(cfg["data"]["root"]).resolve()

    payload = {
        "path": str(data_root),
        "train": "train/images",
        "val": "val/images",
        "nc": 2,
        "names": ["crop", "weed"],
    }
    with open(out_path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
    log.info("YOLO dataset YAML written to %s", out_path)
    return out_path
