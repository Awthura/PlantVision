"""
ONNX export of the trained YOLO segmentation model.
CoreML / TensorRT are explicitly out of scope for this pilot.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def export_onnx(weights_path: str | Path, cfg: dict, opset: int = 17) -> Path:
    """
    Export a trained .pt model to ONNX.

    Parameters
    ----------
    weights_path : path to best.pt produced by training
    cfg          : loaded default.yaml dict
    opset        : ONNX opset version (17 is safe for current ort)

    Returns
    -------
    Path to the exported .onnx file.
    """
    from ultralytics import YOLO

    weights_path = Path(weights_path)
    out_path = Path(cfg["outputs"]["onnx"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Exporting %s → ONNX (opset %d)", weights_path.name, opset)
    model = YOLO(str(weights_path))
    exported = model.export(format="onnx", opset=opset, dynamic=False)
    log.info("ONNX model written to %s", exported)
    return Path(exported)
