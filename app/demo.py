"""
Streamlit demo — upload a UAV image, get overlay + canopy cover %.

Run:
    streamlit run app/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plantvision import segment, traits as trait_fn, visualize

CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_WEIGHTS = Path("models/best.pt")


@st.cache_resource
def load_model(weights: Path, device: str):
    from ultralytics import YOLO
    return YOLO(str(weights))


def main() -> None:
    st.set_page_config(page_title="PlantVision", page_icon="🌱", layout="centered")
    st.title("🌱 PlantVision — Sugar Beet Phenotyping")
    st.caption("Upload a UAV field image to segment crop vs. weed and compute canopy cover %.")

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    with st.sidebar:
        st.header("Settings")
        weights_path = st.text_input("Model weights (.pt)", value=str(DEFAULT_WEIGHTS))
        device = st.selectbox("Device", ["mps", "cpu"], index=0)
        conf = st.slider("Confidence threshold", 0.1, 0.9, float(cfg["infer"]["conf"]), 0.05)
        cfg["infer"]["conf"] = conf

    uploaded = st.file_uploader("Upload field image (PNG / JPG)", type=["png", "jpg", "jpeg"])
    if not uploaded:
        st.info("Waiting for an image…")
        return

    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not decode image.")
        return

    weights = Path(weights_path)
    if not weights.exists():
        st.error(f"Weights not found: {weights}. Train the model first (see README).")
        return

    with st.spinner("Running segmentation…"):
        results = segment.predict_image(weights, uploaded.name, cfg, device=device)
        # predict_image expects a path; pass bytes via temp file
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        results = segment.predict_image(weights, tmp_path, cfg, device=device)
        os.unlink(tmp_path)

    h, w = bgr.shape[:2]
    cover = trait_fn.canopy_cover_from_results(results, cfg["traits"]["crop_class"], (h, w))
    count = trait_fn.plant_count(results, cfg["traits"]["crop_class"])
    overlay = visualize.draw_overlay(bgr, results, cfg)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
    with col2:
        st.image(overlay_rgb, caption="Overlay (green=crop, red=weed)", use_column_width=True)

    st.metric("Canopy cover", f"{cover * 100:.1f}%")
    st.metric("Plant count (estimate)", count)


if __name__ == "__main__":
    main()
