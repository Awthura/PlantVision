"""
Convert PhenoBench instance masks → YOLO segmentation label files.

PhenoBench mask format:
  semantics/     uint16, 0=bg 1=crop 2=weed 3=partial-crop 4=partial-weed
  plant_instances/ uint16, 0=bg, >0 = distinct instance id

YOLO seg label format (one .txt per image, same stem):
  <class> <x1> <y1> <x2> <y2> ... <xn> <yn>   (coords normalised to [0,1])
  class 0 = crop  (semantics 1 or 3)
  class 1 = weed  (semantics 2 or 4)

Labels are written to  <split>/labels/<stem>.txt
alongside the images in <split>/images/<stem>.png

Usage:
    uv run python scripts/convert_to_yolo.py [--data data/PhenoBench] [--workers 4]
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# PhenoBench semantic class → YOLO class
SEM_TO_YOLO = {1: 0, 3: 0, 2: 1, 4: 1}  # crop/partial-crop=0, weed/partial-weed=1

# Polygon simplification: approxPolyDP epsilon as fraction of perimeter
POLY_EPSILON_FRAC = 0.005
MIN_AREA_PX = 50  # skip tiny instances


def instance_to_polygon(binary: np.ndarray, epsilon_frac: float = POLY_EPSILON_FRAC) -> np.ndarray | None:
    """
    Find the largest contour of a binary mask and simplify it.
    Returns Nx2 float32 array of (x, y) in [0,1], or None if unusable.
    """
    h, w = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_AREA_PX:
        return None
    peri = cv2.arcLength(cnt, closed=True)
    eps = epsilon_frac * peri
    approx = cv2.approxPolyDP(cnt, eps, closed=True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    norm = approx.astype(np.float32)
    norm[:, 0] /= w
    norm[:, 1] /= h
    norm = np.clip(norm, 0.0, 1.0)
    return norm


def convert_image(args: tuple) -> int:
    """Worker: convert one image. Returns number of label lines written."""
    img_path, labels_dir = args
    sem_path = img_path.parent.parent / "semantics" / img_path.name
    inst_path = img_path.parent.parent / "plant_instances" / img_path.name

    if not sem_path.exists() or not inst_path.exists():
        return 0

    sem  = np.array(Image.open(sem_path))   # uint16
    inst = np.array(Image.open(inst_path))  # uint16

    lines: list[str] = []
    for inst_id in np.unique(inst):
        if inst_id == 0:
            continue
        mask = (inst == inst_id).astype(np.uint8)
        # determine YOLO class from majority semantics vote inside instance
        sem_vals = sem[mask == 1]
        if sem_vals.size == 0:
            continue
        values, counts = np.unique(sem_vals, return_counts=True)
        best_sem = int(values[np.argmax(counts)])
        yolo_cls = SEM_TO_YOLO.get(best_sem)
        if yolo_cls is None:
            continue
        poly = instance_to_polygon(mask)
        if poly is None:
            continue
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
        lines.append(f"{yolo_cls} {coords}")

    label_path = labels_dir / (img_path.stem + ".txt")
    label_path.write_text("\n".join(lines))
    return len(lines)


def convert_split(split_dir: Path, workers: int) -> None:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    images = sorted(images_dir.glob("*.png"))
    log.info("[%s] converting %d images with %d workers…", split_dir.name, len(images), workers)

    args = [(p, labels_dir) for p in images]
    with mp.Pool(workers) as pool:
        counts = pool.map(convert_image, args)

    total_instances = sum(counts)
    skipped = sum(1 for c in counts if c == 0)
    log.info(
        "[%s] done — %d label files, %d total instances, %d empty/skipped",
        split_dir.name, len(images), total_instances, skipped,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/PhenoBench")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()

    root = Path(args.data)
    for split in ("train", "val"):
        convert_split(root / split, args.workers)
    log.info("Conversion complete. Labels written to <split>/labels/")


if __name__ == "__main__":
    main()
