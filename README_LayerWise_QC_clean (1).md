![CI](https://github.com/MohammadAminNouri/LayerWise-QC/actions/workflows/ci.yml/badge.svg)

# LayerWise-QC

LayerWise-QC is a prototype for layer-wise quality monitoring in laser powder-bed fusion. It combines process parameters, sensor-image channels, defect-risk logic, and a live dashboard in one small research-oriented software package.

The project is built around a simple question:

```text
Can each printed layer be checked while the build is still running,
using the information already available from the machine and its sensors?
```

The current version supports three sensing paths:

```text
OT   = optical tomography / thermal emission image
MPM  = melt-pool monitoring image or rasterized melt-pool signal
PBI  = powder-bed image before or after exposure
```

The software can work with one channel, but the main structure is multimodal: one model reads optical/thermal process information, another model reads a second sensor channel, and their predictions are combined into a fused layer-quality score.

The included data is only a demo set. It is generated to make the code and interface runnable without laboratory data. Real OT, MPM, or powder-bed images can be connected later through the manifest file.

---

## Main idea

```text
layer image
+ process parameters
+ sensor-specific model
+ fusion of model outputs
+ process-aware risk score
= layer quality estimate
```

The project is not a finished industrial controller. It is a working prototype for testing the structure of an in-situ quality-monitoring pipeline.

---

## What the project does

- Calculates volumetric energy density from laser power, scan speed, hatch distance, and layer thickness.
- Organizes layer-wise image patches using a CSV manifest.
- Trains one image model per sensor channel.
- Supports OT, MPM, and PBI paths.
- Uses weighted sampling for imbalanced defect classes.
- Applies image augmentation to imitate sensor noise, small shifts, and acquisition variation.
- Combines two model outputs through late-fusion probability scoring.
- Reports metrics that are useful for imbalanced data: MCC, balanced accuracy, weighted F1, precision, and recall.
- Generates Grad-CAM maps to inspect what image regions influence the model.
- Provides a Streamlit dashboard for live demonstration.
- Includes a small demo dataset so the repository works immediately after installation.

---

## Folder layout

```text
app/
  streamlit_app.py              # live dashboard
configs/
  default.yaml                  # training and experiment settings
data/
  demo_samples/                 # generated demo patches and manifest
  README.md
scripts/
  make_synthetic_dataset.py     # creates a runnable synthetic dataset
  train_modality.py             # trains OT, MPM, or PBI model
  evaluate_ensemble.py          # fuses OT with a second sensor channel
  run_gradcam.py                # saves Grad-CAM overlays
  run_full_demo.py              # quick command-line demonstration
src/am_defect_detection/
  constants.py
  data.py
  gradcam.py
  metrics.py
  models.py
  simulation.py
  training.py
  transforms.py
  utils.py
tests/
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

---

## Run the live dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard is the easiest way to show the project. It contains:

### 1. Process console

Change:

```text
laser power
scan speed
hatch distance
layer thickness
heat memory
powder uniformity
fusion weights
```

The dashboard recalculates VED and updates the estimated quality/risk score.

### 2. Sample sensor patches

The app shows built-in OT, MPM, and PBI demo patches for normal and disturbed conditions. These are not experimental measurements. They are safe demo images included only to make the interface understandable.

### 3. Layer story

A simple layer-by-layer view shows how a disturbed region may appear and how the risk changes when the process returns toward stable parameters.

---

## Quick demo from terminal

```bash
python scripts/run_full_demo.py --epochs 1 --layers 40 --no-pretrained
```

This command creates a small dataset, trains two sensor-specific models, fuses their outputs, and saves a Grad-CAM example.

Use `--no-pretrained` when running offline or when torchvision cannot download model weights.

---

## Create a synthetic dataset

```bash
python scripts/make_synthetic_dataset.py \
  --out data/synthetic \
  --layers 172 \
  --modalities ot,mpm,pbi
```

The synthetic dataset uses:

```text
30 specimens
3 process states
layer-wise image patches
standard / low-energy / high-energy class labels
```

It is useful for checking the pipeline, debugging the dashboard, and testing code changes. It should not be used as proof of real defect-detection performance.

---

## Train one model per sensor channel

OT model:

```bash
python scripts/train_modality.py \
  --manifest data/synthetic/manifest.csv \
  --modality ot \
  --out outputs/ot_run \
  --epochs 30
```

MPM model:

```bash
python scripts/train_modality.py \
  --manifest data/synthetic/manifest.csv \
  --modality mpm \
  --out outputs/mpm_run \
  --epochs 30
```

PBI model:

```bash
python scripts/train_modality.py \
  --manifest data/synthetic/manifest.csv \
  --modality pbi \
  --out outputs/pbi_run \
  --epochs 30
```

---

## Fuse two channels

OT + MPM:

```bash
python scripts/evaluate_ensemble.py \
  --manifest data/synthetic/manifest.csv \
  --ot-checkpoint outputs/ot_run/best.pt \
  --second-checkpoint outputs/mpm_run/best.pt \
  --second-modality mpm \
  --out outputs/ensemble_ot_mpm
```

OT + PBI:

```bash
python scripts/evaluate_ensemble.py \
  --manifest data/synthetic/manifest.csv \
  --ot-checkpoint outputs/ot_run/best.pt \
  --second-checkpoint outputs/pbi_run/best.pt \
  --second-modality pbi \
  --out outputs/ensemble_ot_pbi
```

Default fusion:

```text
OT weight      = 0.5
second weight  = 0.5
```

The weights can be changed:

```bash
--w-ot 0.6 --w-second 0.4
```

---

## Grad-CAM

```bash
python scripts/run_gradcam.py \
  --manifest data/synthetic/manifest.csv \
  --checkpoint outputs/ot_run/best.pt \
  --modality ot \
  --row 0 \
  --out outputs/gradcam
```

The saved overlay shows which part of the image patch influenced the prediction.

This is useful because sensor-based monitoring should not only produce a class label. The model should also be inspected to see whether it is reacting to meaningful regions or to irrelevant image artifacts.

---

## Using real data

Create a CSV manifest with this structure:

```text
sample_id,layer,specimen_id,class_idx,class_name,ot_path,mpm_path,pbi_path,laser_power_w,scan_speed_mm_s,hatch_distance_mm,layer_thickness_mm,ved_j_mm3
```

Minimum required columns for training are:

```text
sample_id,class_idx,class_name,<modality>_path
```

Example for PBI training:

```text
sample_id,class_idx,class_name,pbi_path
```

Relative paths are resolved from the manifest folder. Absolute paths also work.

---

## Labels used in the demo

```text
0 = standard
1 = delta_minus_30_ved       # low-energy / lack-of-fusion risk
2 = delta_plus_30_ved        # high-energy / overheating/keyhole-spatter risk
```

The class names can be replaced when real labels are available, for example:

```text
good
lack_of_fusion
recoater_defect
overheated
porosity
surface_irregularity
```

---

## Why this structure is useful

Layer-wise process monitoring has two problems at the same time:

```text
The physics matters.
The sensor data is visual and noisy.
```

This project keeps both sides in the same workflow.

The image models handle sensor patterns. The process console handles simple physical indicators such as VED and parameter movement. The fusion step keeps the channels separate until the final score, which makes the system easier to inspect and extend.

This is especially useful for a thesis prototype because each part can be improved independently:

```text
better sensor preprocessing
better image model
better fusion method
better physical risk model
better validation against CT/metallography/density results
```

---

## Current limits

The current repository does not prove real defect-detection performance. It needs experimental data for that.

A proper validation study would need:

- real layer-wise OT images,
- real MPM or powder-bed images,
- correct layer and patch alignment,
- ground-truth labels from CT, metallography, density measurement, or another inspection method,
- a train/validation/test split that avoids leakage between similar neighboring patches,
- testing on builds that were not used during training.

The demo data is only for software testing and presentation.

---

## Tests

```bash
pytest -q
```

Current tests check:

```text
VED calculation
late-fusion probability logic
```

---

## Suggested next steps

- Replace demo images with real OT and PBI patches.
- Add layer-history features, not only single-layer inputs.
- Add a stronger physical risk model using heat accumulation or scan-strategy information.
- Compare OT-only, PBI-only, and OT+PBI performance.
- Use Grad-CAM outputs to check whether the model focuses on meaningful process regions.
- Validate predictions against CT or metallography.
