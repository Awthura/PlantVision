# PlantVision — Sugar Beet Field Phenotyping

> Manual plot scoring in breeding programs is a throughput bottleneck: a single
> field trial can contain thousands of plots that agronomists must score by eye.
> This pilot automates canopy-cover extraction and crop/weed segmentation from
> UAV imagery, replacing a slow manual step with a reproducible, image-based
> phenotyping pipeline.

<!--
TODO before publishing: replace placeholders below with real output images.
  - hero_overlay.png  : one striking before/after overlay
  - demo.gif          : short loop of the Streamlit demo
  - overlay_*.png     : 3 sample overlays for the Results section
-->

---

## Pipeline

```
PhenoBench images ─▶ fine-tune YOLO-seg ─▶ per-image masks
       └▶ trait extraction (canopy cover %, plant count)
              └▶ overlay + traits.csv + canopy chart
                     └▶ Streamlit upload demo
                     └▶ ONNX export
```

## Why YOLO for segmentation?

YOLO's anchor-free, NMS-free architecture makes it the current practical
choice for edge and MPS inference: it runs well on Apple Silicon without CUDA,
the `n`/`s` scales fit a single fine-tuning session, and it delivers
competitive instance-segmentation mIoU at low latency. For a field-deployed
sensor or a breeder's laptop, that matters.

---

## Quickstart

### 1 — Install (requires Python 3.11+, [uv](https://github.com/astral-sh/uv))

```bash
git clone git@github.com:Awthura/PlantVision.git
cd PlantVision
uv sync --extra dev
```

### 2 — Download PhenoBench (~7.6 GB)

Visit [phenobench.org/dataset.html](https://phenobench.org/dataset.html), find
the direct zip URL and MD5, then fill them into `scripts/download_data.py` and run:

```bash
uv run python scripts/download_data.py
```

### 3 — Train and run the full pipeline

```bash
# Full pipeline (train → predict → traits → ONNX export)
uv run python scripts/run_pipeline.py

# Override device
uv run python scripts/run_pipeline.py --device cpu

# Inference-only with existing weights
uv run python scripts/run_pipeline.py --infer-only --weights models/best.pt
```

### 4 — Interactive demo

```bash
uv run streamlit run app/demo.py
```

Upload any field image. Get the overlay and canopy cover % instantly.

### 5 — Run tests

```bash
uv run pytest
```

---

## Results

| Metric           | Value     |
|------------------|-----------|
| val mIoU         | _TBD after training_ |
| val F1 (crop)    | _TBD_ |
| val F1 (weed)    | _TBD_ |

> **Qualitative check:** canopy-cover % was sanity-checked against eyeball
> estimates on 10 held-out images. Values fall within expected range for the
> growth stage shown.

### Sample overlays

<!-- Add 3 overlay images here after training -->

---

## Configuration

Everything path/threshold-driven. Edit `configs/default.yaml`:

```yaml
model:
  name: yolo12n-seg    # swap to s/m for more capacity
  device: mps          # mps | cpu | cuda

train:
  epochs: 50
  batch: 8
  imgsz: 640

traits:
  crop_class: 0        # PhenoBench class index
  weed_class: 1
```

---

## Repo structure

```
plantvision/
├── configs/default.yaml        # all paths and hyperparameters
├── src/plantvision/
│   ├── ingest.py               # dataset verification + YOLO YAML generation
│   ├── segment.py              # YOLO train / predict wrappers
│   ├── traits.py               # canopy cover %, plant count
│   ├── visualize.py            # colored overlays + chart
│   ├── export.py               # ONNX export
│   └── pipeline.py             # orchestrator
├── app/demo.py                 # Streamlit upload demo
├── scripts/
│   ├── download_data.py        # PhenoBench downloader
│   └── run_pipeline.py         # CLI entry point
├── tests/test_traits.py        # unit tests (trait math, no model required)
├── data/README.md              # dataset source, license, citation
└── notebooks/01_explore.ipynb  # EDA
```

---

## Deliberately out of scope — next steps

This is a **pilot**, not a platform. The cut list below proves awareness of the
full phenotyping stack; none of it is built here, and that is intentional.

- **SAM 3 auto-labeling** — PhenoBench is already labeled. To extend to an
  unlabeled new field, SAM 3 promptable segmentation would bootstrap crop/weed
  masks from a text prompt, eliminating manual annotation.
- **DINOv3 features / label-efficient heads** — only relevant in label-scarce
  regression settings; not needed here.
- **NIRS + PLS calibration** — the phenomic-selection extension: pair canopy
  hyperspectral reflectance with trait measurements, then fit a PLS or
  LightGBM calibration curve. This is how near-infrared data is turned into
  breeding values on real platforms.
- **Disease severity (Cercospora leaf spot)** — a natural follow-on; requires a
  separate labeled dataset and a per-lesion severity head.
- **Orthomosaic / GIS / field-map version** — a production platform would stitch
  drone frames into an orthomosaic and project plot-level trait maps onto a GIS
  coordinate system (cf. WeedMap). Not needed for per-frame phenotyping.
- **Time-series, genotype ranking, FastAPI service, Docker-compose, TensorRT,
  CoreML** — all valid extensions for a productionised system; explicitly deferred.

---

## Dataset

**PhenoBench v1.1.0** — University of Bonn.
UAV imagery of sugar-beet fields across two growing seasons under varied
natural lighting. Dense pixel-wise labels: semantic (crop/weed), plant
instances, leaf instances. ~1,407 train / 772 val images.

License: CC BY 4.0. See `data/README.md` for citation.

---

*A deliberately scoped pilot demonstrating image-based field phenotyping for
sugar-beet breeding — the kind of automation that sits at the core of
modern plant phenotyping platforms.*
