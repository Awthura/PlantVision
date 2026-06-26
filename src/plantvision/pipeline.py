"""
End-to-end pipeline: ingest → train → predict → traits → visualize → export.
Each stage is independently callable; run_all() runs the full sequence.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import pandas as pd

from plantvision import ingest, segment, traits as trait_fn, visualize, export

log = logging.getLogger(__name__)


def run_ingest(cfg: dict) -> None:
    ingest.verify_split(Path(cfg["data"]["train"]), "train")
    ingest.verify_split(Path(cfg["data"]["val"]), "val")
    stats = ingest.dataset_stats(cfg)
    log.info("Dataset stats: %s", stats)
    ingest.build_yolo_dataset_yaml(cfg)


def run_train(cfg: dict, device: str | None = None) -> Path:
    best = segment.train(cfg, device=device)
    return best


def run_predict_and_traits(
    weights: Path,
    cfg: dict,
    device: str | None = None,
    max_images: int | None = None,
) -> pd.DataFrame:
    """Run inference on val split, extract traits, save overlays. Returns traits DF."""
    crop_cls = cfg["traits"]["crop_class"]
    overlay_dir = Path(cfg["outputs"]["overlays"])
    records = []

    val_images = sorted(Path(cfg["data"]["val"]).glob("images/*.png"))
    if max_images:
        val_images = val_images[:max_images]

    for img_path in val_images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            log.warning("Could not read %s", img_path)
            continue

        results = segment.predict_image(weights, img_path, cfg, device=device)
        h, w = bgr.shape[:2]
        cover_pct = round(trait_fn.canopy_cover_from_results(results, crop_cls, (h, w)) * 100, 2)
        count = trait_fn.plant_count(results, crop_cls)

        overlay = visualize.draw_overlay(bgr, results, cfg)
        visualize.save_overlay(img_path, overlay, overlay_dir)

        records.append(
            {
                "image": img_path.name,
                "canopy_cover_pct": cover_pct,
                "plant_count": count,
            }
        )
        log.info("%s  cover=%.1f%%  count=%d", img_path.name, cover_pct, count)

    df = pd.DataFrame(records)
    csv_path = Path(cfg["outputs"]["traits"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    log.info("Traits saved to %s", csv_path)
    return df


def run_visualize(df: pd.DataFrame, cfg: dict) -> None:
    visualize.plot_canopy_chart(df, Path(cfg["outputs"]["chart"]))


def run_export(weights: Path, cfg: dict) -> Path:
    return export.export_onnx(weights, cfg)


def run_all(cfg: dict, device: str | None = None) -> None:
    log.info("=== PlantVision pipeline start ===")
    run_ingest(cfg)
    weights = run_train(cfg, device=device)
    df = run_predict_and_traits(weights, cfg, device=device)
    run_visualize(df, cfg)
    run_export(weights, cfg)
    log.info("=== Pipeline complete. See outputs/ ===")
