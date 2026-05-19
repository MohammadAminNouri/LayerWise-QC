# In-situ AM quality monitor

A working prototype for layer-wise quality monitoring in laser powder-bed fusion. It was made to test a similar line of work to the OT/MPM defect-detection study, but the code is kept a bit broader: it supports **OT + MPM** and **OT + powder-bed imaging (PBI)**, and it also has a small live dashboard for explaining the idea without needing the full lab dataset.

The main idea is simple:

```text
sensor image from each layer
+ process parameters such as P, v, h, L
+ one model per sensing channel
+ late fusion of the output probabilities
= layer/region quality score
```

This repository does **not** contain private experimental data. The included images are small simulated patches just to make the software runnable and easy to show. Real OT, MPM, or powder-bed images can be dropped in through the manifest file.

## What is inside

- VED calculation from laser power, scan speed, hatch distance and layer thickness.
- Patch-based dataset structure for layer-wise OT / MPM / PBI images.
- ResNet-18 modality models using transfer learning.
- Weighted sampling for class imbalance.
- Image augmentations for sensor noise and small misalignment.
- Late-fusion soft voting: OT model + second sensor model.
- Metrics suited to imbalanced data: MCC, balanced accuracy, weighted F1, precision, recall.
- Grad-CAM output for checking what the model is using.
- Streamlit app to play with parameters and display sample sensor patches.
- A tiny built-in demo set so the interface opens immediately.

## Folder layout

```text
app/
  streamlit_app.py              # live dashboard
configs/
  default.yaml                  # experiment and training settings
data/
  demo_samples/                 # 9 safe demo images + small manifest
  README.md
scripts/
  make_synthetic_dataset.py     # creates a runnable synthetic dataset
  train_modality.py             # train OT, MPM, or PBI model
  evaluate_ensemble.py          # fuse OT + second channel
  run_gradcam.py                # save a Grad-CAM overlay
  run_full_demo.py              # quick command-line demo
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

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

## Open the live dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard has three parts:

1. **live console** — change laser power, scan speed, hatch distance, layer thickness, heat-memory and powder uniformity. It recalculates VED and shows a fused quality score.
2. **sample images** — shows built-in OT, MPM, and PBI sample patches for the three classes.
3. **layer story** — a simple layer-by-layer sketch of a disturbed region and the self-healing idea.

The app does not claim to be a real-time controller. It is a small front-end for discussion and demonstration. The trained-model part can be connected later once real sensor data and checkpoints are available.

## Run a quick command-line demo

```bash
python scripts/run_full_demo.py --epochs 1 --layers 40 --no-pretrained
```

This creates a small synthetic dataset, trains an OT model and an MPM model, fuses the predictions, and saves a Grad-CAM image. Use `--no-pretrained` when running offline or when torchvision cannot download weights.

## Make a larger synthetic dataset

```bash
python scripts/make_synthetic_dataset.py \
  --out data/synthetic \
  --layers 172 \
  --modalities ot,mpm,pbi
```

This follows the same basic layout used in the prototype: 30 specimens, standard / low-VED / high-VED rows, and a notch region with increasing disturbed layers. The images are still simulated, not measurements.

## Train one model per sensor

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

## Fuse OT with the second channel

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

The default fusion weights are 0.5 and 0.5. They can be changed:

```bash
--w-ot 0.6 --w-second 0.4
```

## Grad-CAM

```bash
python scripts/run_gradcam.py \
  --manifest data/synthetic/manifest.csv \
  --checkpoint outputs/ot_run/best.pt \
  --modality ot \
  --row 0 \
  --out outputs/gradcam
```

The result is an overlay image showing which part of the patch influenced the classifier.

## Using real lab data

Create a CSV manifest like this:

```text
sample_id,layer,specimen_id,class_idx,class_name,ot_path,mpm_path,pbi_path,laser_power_w,scan_speed_mm_s,hatch_distance_mm,layer_thickness_mm,ved_j_mm3
```

Minimum required columns for training are:

```text
sample_id,class_idx,class_name,<modality>_path
```

For example, to train PBI:

```text
sample_id,class_idx,class_name,pbi_path
```

Relative paths are resolved from the manifest folder by default. Absolute paths also work.

## Classes used here

```text
0 = standard
1 = delta_minus_30_ved       # low-energy / lack-of-fusion risk
2 = delta_plus_30_ved        # high-energy / keyhole-spatter risk
```

The names are kept close to the experimental logic, but they can be renamed if the real dataset uses labels such as `good`, `recoater_defect`, `lack_of_fusion`, `overheated`, or `porosity`.

## How this is different from only copying the earlier work

The earlier work is treated as the starting point: two sensing channels, modality-specific CNNs, and late fusion. This prototype adds a few practical extensions:

- PBI is implemented as a second channel next to MPM.
- The manifest can hold OT + MPM + PBI at the same time.
- The Streamlit app shows process-parameter effects, VED movement and sensor examples live.
- A simple process-aware score is included in the app so the project is not only an image classifier.
- The code keeps a path open for OT + PBI, which is closer to the possible thesis direction.

## What still needs real data

The current demo cannot prove defect detection performance. For that, the project needs:

- real layer-wise OT images,
- real MPM or powder-bed images,
- layer/patch alignment,
- ground-truth labels from metallography, CT, density, or another inspection method,
- train/validation/test split that avoids leakage between near-identical patches.

Once those are available, the same scripts can be used without changing the whole structure.

## Tests

```bash
pytest -q
```

Current tests check the VED calculation and the late-fusion probability logic.
