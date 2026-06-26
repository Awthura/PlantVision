"""
Entry point for the full PlantVision pipeline.

Usage:
    # Full run
    python scripts/run_pipeline.py

    # Inference-only (skip training, use existing weights)
    python scripts/run_pipeline.py --infer-only --weights models/best.pt

    # Override device
    python scripts/run_pipeline.py --device cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from plantvision import pipeline
from plantvision.ingest import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="PlantVision pipeline runner")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None, help="mps | cpu | cuda")
    parser.add_argument("--infer-only", action="store_true", help="Skip training")
    parser.add_argument("--weights", default=None, help="Path to .pt for --infer-only")
    parser.add_argument("--max-images", type=int, default=None, help="Limit val images for quick test")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.infer_only:
        weights = Path(args.weights or cfg["outputs"]["models"] + "/best.pt")
        if not weights.exists():
            parser.error(f"--infer-only requires --weights pointing to a .pt file. Got: {weights}")
        df = pipeline.run_predict_and_traits(weights, cfg, device=args.device, max_images=args.max_images)
        pipeline.run_visualize(df, cfg)
        pipeline.run_export(weights, cfg)
    else:
        pipeline.run_all(cfg, device=args.device)


if __name__ == "__main__":
    main()
