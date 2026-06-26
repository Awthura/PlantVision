# PlantVision — Sugar Beet Field Phenotyping (PILOT)

> **Build spec for Claude Code. This is a deliberately bounded PILOT.**
> Build exactly what is in §3. Stop at the line in §4. Everything in §5 is
> **described in the README as future work but NOT built.** Before pinning
> versions, verify current package versions and the dataset download. Do not
> invent links.

---

## 1. What this is (one paragraph)

A small, focused computer-vision pilot: take public UAV imagery of a sugar-beet
field, segment crop vs. weed, and compute a phenotypic trait (canopy cover %,
optionally plant count) per image, with a clean overlay, a tiny demo app, and a
portfolio-grade README. It is a demonstrator + application artifact for a Data
Scientist role in sugar-beet breeding. Built *like* a production pipeline
(modular, config-driven) but scoped to a single weekend.

---

## 2. The one rule

This is a **pilot**, not a platform. The goal is a clean, complete, showable
thing — not coverage of the entire phenotyping stack. A tight project with an
honest "what I left out and why" section reads as **good engineering judgment**.
The cut list in §5 is the backbone of the project, not a TODO list. Resist scope
creep.

---

## 3. Build this — the pilot

**Dataset (locked): PhenoBench.** See §7.

**Pipeline:**
```
PhenoBench images ─▶ fine-tune YOLO26-seg ─▶ per-image masks
       └▶ trait extraction (canopy cover %, opt. plant count)
              └▶ overlay + per-image trait CSV + small chart
                     └▶ tiny Streamlit/Gradio demo
                     └▶ ONNX export of the model
```

**Steps:**
1. **Ingest** — load PhenoBench train/val; basic dataset stats; config-driven paths.
2. **Train** — fine-tune `yolo26n-seg` (or `s`) on PhenoBench crop/weed masks.
3. **Trait extraction** —
   - **Must-have:** canopy cover % = crop-mask pixels ÷ image pixels.
   - **Bonus (only if quick):** plant/stand count via crop *instances*.
4. **Visualize** — colored overlay (crop green / weed red) + a per-image trait
   value; write a `traits.csv`.
5. **Demo** — minimal Streamlit/Gradio app: upload an image → overlay + trait number.
6. **Export** — ONNX export of the trained model (one short module). CoreML
   optional. TensorRT explicitly **not** in scope (no CUDA on the target Mac).
7. **README** — the real deliverable (see §10).

---

## 4. ⛔ STOP HERE

The pilot is **DONE** when you have:
- a trained YOLO26-seg model with a reported val mIoU,
- canopy-cover overlays + a `traits.csv`,
- a working upload demo,
- an ONNX export,
- a polished README.

**Do not start anything in §5.** If you find yourself reaching for a second
dataset, a second modality, or a "while I'm here" feature — stop. That's the
scope-creep signal.

---

## 5. Deliberately OUT OF SCOPE (describe in README, do NOT build)

Each of these gets **one or two sentences** in the README's "Next steps /
deliberately out of scope" section. Building none of them is the point; naming
them proves you see the full landscape and made a scoping decision.

- **SAM 3 auto-labeling** — not needed; PhenoBench is already labeled. Frame as:
  "to extend to an unlabeled new field, SAM 3 promptable concept segmentation
  would bootstrap masks from a text prompt."
- **DINOv3 features / label-efficient heads** — only relevant for label-scarce
  regression; not needed here.
- **NIRS + PLS calibration** — the phenomic-selection extension. Describe it
  (canopy/spectra → trait via PLS/LightGBM) to show you understand the
  calibration side; do not build it.
- **Disease severity (Cercospora)** — a separate dataset and task; mention as a
  natural follow-on.
- **Orthomosaic / GIS / WeedMap field-map version** — note awareness (drone →
  orthomosaic → plot map is how a real platform is structured); do not build geo
  plumbing.
- **Time-series, genotype ranking, FastAPI service, Docker-compose, TensorRT/
  CoreML** — all future work.

---

## 6. Tech & hardware (kept tight)

- Python 3.11+, **uv** for env (lockfile committed).
- PyTorch on **Apple-Silicon MPS** (`--device mps|cpu`), small `yolo26n/s` scales.
- `ultralytics` for YOLO26. Verify the current version/model name before pinning.
- `pytest` for trait math only. Optional plain `Dockerfile` (cheap credibility
  signal; skip if time-pressed).
- Everything path/threshold lives in `configs/default.yaml`. No hardcoded paths.

---

## 7. Dataset (locked): PhenoBench

University of Bonn, UAV imagery of **sugar-beet fields** under natural lighting,
multiple days across two years (varied growth stages + lighting). Dense
pixel-wise masks: semantic (crop/weed), plant instances, and leaf instances.

- Download: `PhenoBench-v110.zip` (~7.6 GB) from the official dataset page
  (`phenobench.org/dataset.html`). **Confirm the URL/checksum at build time.**
- Size: ~1,407 train + 772 val images; labels include `semantics`,
  `plant_instances`, `leaf_instances`.
- Note: PhenoBench ships **original UAV frames, not an orthomosaic** — that's
  fine and intended for this pilot.
- `scripts/download_data.py` fetches into `data/`; `data/README.md` records
  source, license, and citation.

---

## 8. Repo structure (trimmed)

```
plantvision/
├── README.md                 # portfolio-grade (§10)
├── pyproject.toml            # uv; pinned
├── Dockerfile                # optional
├── configs/default.yaml
├── data/                     # gitignored; populated by download script
│   └── README.md
├── src/plantvision/
│   ├── ingest.py
│   ├── segment.py            # YOLO26 train / predict
│   ├── traits.py             # canopy cover %, optional count
│   ├── visualize.py          # overlays + chart
│   ├── export.py             # ONNX export
│   └── pipeline.py
├── app/demo.py               # Streamlit/Gradio
├── scripts/{download_data.py,run_pipeline.py}
├── tests/                    # trait-math unit tests
└── notebooks/01_explore.ipynb
```

---

## 9. Evaluation (light)

- Segmentation: val **mIoU / F1** from the YOLO26 run.
- Traits: sanity-check canopy cover on a few images (eyeball overlay vs. value).
- Qualitative: **3 before/after overlays + 1 short GIF** for the README.
- Reproducible: one `uv run` command reproduces the core result on CPU/MPS.

---

## 10. README (the real deliverable)

- 2-sentence problem statement in **breeding** terms (manual plot scoring is the
  bottleneck; image-based phenotyping automates it).
- Hero overlay image + GIF near the top.
- Short **"Why YOLO26"** note (NMS-free, edge-friendly, small-scale, current SOTA).
- Architecture diagram (reuse §3).
- Quickstart: clone → install → download → run → result.
- Results: mIoU + qualitative figures.
- **"Deliberately out of scope / next steps"** section (from §5) — this is a
  feature, not an apology.
- Closing line tying it to real breeding phenotyping platforms.

---

## 11. Instructions to Claude Code

1. **Verify** the `ultralytics`/YOLO26 version and the PhenoBench download
   (URL + checksum) before pinning anything.
2. Scaffold the repo, commit, then build §3 **in order**.
3. **Stop at §4.** Do not implement anything from §5; instead write it into the
   README "next steps" section.
4. Config-driven, no hardcoded paths. MPS default with a `--device` flag.
5. Write unit tests for the trait math (synthetic masks).
6. Keep dependencies lean.

---

## 12. Time-box

Target ~one weekend / 3–4 evenings. If a step balloons, **cut the bonus plant
count before cutting quality** on canopy cover, the demo, or the README. A small
thing done cleanly beats a big thing left half-finished.

---

*Author: [your name]. A deliberately scoped pilot demonstrating image-based field
phenotyping for sugar-beet breeding.*
