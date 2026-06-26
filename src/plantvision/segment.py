"""
YOLO segmentation — training and inference wrappers.

Model name verified: yolo26n-seg.pt / yolo26s-seg.pt (ultralytics 8.4.79).
Swap cfg.model.name to yolo26s-seg for higher mIoU at the cost of speed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def train(cfg: dict, device: str | None = None) -> Path:
    """Fine-tune YOLO seg on PhenoBench. Returns path to best weights."""
    from ultralytics import YOLO  # imported lazily so non-GPU envs can import module

    model_name: str = cfg["model"]["name"]
    dev = device or cfg["model"]["device"]
    t = cfg["train"]

    dataset_yaml = Path("configs/phenobench_yolo.yaml")
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"{dataset_yaml} not found. Run ingest.build_yolo_dataset_yaml() first."
        )

    log.info("Training %s on %s for %d epochs", model_name, dev, t["epochs"])
    model = YOLO(f"{model_name}.pt")
    results = model.train(
        data=str(dataset_yaml),
        epochs=t["epochs"],
        batch=t["batch"],
        imgsz=t["imgsz"],
        lr0=t["lr0"],
        patience=t["patience"],
        workers=t["workers"],
        device=dev,
        project=t["project"],
        name=t["name"],
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    log.info("Training complete. Best weights: %s", best)
    return best


def predict_image(model_path: str | Path, image_path: str | Path, cfg: dict, device: str | None = None):
    """Run inference on a single image. Returns ultralytics Results object."""
    from ultralytics import YOLO

    dev = device or cfg["model"]["device"]
    model = YOLO(str(model_path))
    results = model.predict(
        source=str(image_path),
        conf=cfg["infer"]["conf"],
        iou=cfg["infer"]["iou"],
        device=dev,
        verbose=False,
    )
    return results[0]


def predict_batch(model_path: str | Path, image_dir: str | Path, cfg: dict, device: str | None = None):
    """Yield (image_path, Results) for all PNGs in image_dir."""
    from ultralytics import YOLO

    dev = device or cfg["model"]["device"]
    model = YOLO(str(model_path))
    images = sorted(Path(image_dir).glob("*.png"))

    for img_path in images:
        results = model.predict(
            source=str(img_path),
            conf=cfg["infer"]["conf"],
            iou=cfg["infer"]["iou"],
            device=dev,
            verbose=False,
        )
        yield img_path, results[0]
