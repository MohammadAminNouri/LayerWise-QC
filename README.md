
![CI](https://github.com/MohammadAminNouri/LayerWise-QC/actions/workflows/ci.yml/badge.svg)

LayerWise-QC is a small research prototype for in-situ quality monitoring in laser powder-bed fusion. It connects process parameters, sensor-image channels, and a live dashboard to estimate how stable or risky a printed layer may be.

Live app: https://layerwise-qc-nouri.streamlit.app/  
Repository: https://github.com/MohammadAminNouri/LayerWise-QC

## What it does

During laser powder-bed fusion, each printed layer is affected by the energy input, the powder-bed condition, and the melt-zone response. LayerWise-QC puts these pieces into one simple workflow:

```text
laser parameters + layer image channels -> sensor scores -> fused layer-quality estimate
```

The current version is built for demonstration and development. It uses generated sample patches so the dashboard and scripts can run without private laboratory data. The same structure can later be connected to real optical tomography, melt-pool monitoring, or powder-bed imaging data through a CSV manifest.

## Dashboard

Run the app with:

```bash
streamlit run app/streamlit_app.py
```

The dashboard is the main entry point. It lets the user change laser power, scan speed, hatch distance, layer thickness, powder uniformity, heat-memory, and fusion weights. Every change updates the volumetric energy density, the sensor-channel scores, and the final risk interpretation.

The dashboard is useful because it shows the relation between process settings and the quality estimate in one place. Lowering scan speed increases energy input. Increasing scan speed reduces energy input. Poorer powder uniformity increases the powder-bed risk contribution. Changing fusion weights shows how much the final output depends on the optical channel or on the second sensor channel.

## Software structure

```text
app/
  streamlit_app.py              # interactive dashboard
configs/
  default.yaml                  # default experiment settings
data/
  demo_samples/                 # generated sample patches used by the app
  README.md
scripts/
  make_synthetic_dataset.py     # creates a runnable synthetic dataset
  train_modality.py             # trains one model for OT, MPM, or PBI
  evaluate_ensemble.py          # combines two sensor models
  run_gradcam.py                # creates Grad-CAM overlays
  run_full_demo.py              # short end-to-end command-line run
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

## Main components

The code includes a volumetric energy density calculation, a manifest-based image dataset, a ResNet-style image model for each sensor channel, weighted sampling for imbalanced classes, basic image augmentation, late-fusion scoring, imbalanced-data metrics, and Grad-CAM output for checking model attention.

The intended sensor paths are:

```text
OT   optical tomography or thermal emission image
MPM  melt-pool monitoring image or rasterized signal
PBI  powder-bed image before or after exposure
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
streamlit run app/streamlit_app.py
```

## Run a short demo

```bash
python scripts/run_full_demo.py --epochs 1 --layers 40 --no-pretrained
```

This creates a small generated dataset, trains two sensor models, fuses their predictions, and saves one Grad-CAM example. The `--no-pretrained` option is useful when running offline.

## Create a larger generated dataset

```bash
python scripts/make_synthetic_dataset.py \
  --out data/synthetic \
  --layers 172 \
  --modalities ot,mpm,pbi
```

The generated dataset is only for checking the pipeline and dashboard behavior. It is not a substitute for real process data.

## Train sensor models

```bash
python scripts/train_modality.py \
  --manifest data/synthetic/manifest.csv \
  --modality ot \
  --out outputs/ot_run \
  --epochs 30

python scripts/train_modality.py \
  --manifest data/synthetic/manifest.csv \
  --modality pbi \
  --out outputs/pbi_run \
  --epochs 30
```

## Fuse two channels

```bash
python scripts/evaluate_ensemble.py \
  --manifest data/synthetic/manifest.csv \
  --ot-checkpoint outputs/ot_run/best.pt \
  --second-checkpoint outputs/pbi_run/best.pt \
  --second-modality pbi \
  --out outputs/ensemble_ot_pbi
```

Default fusion uses equal weights. The weights can be changed:

```bash
--w-ot 0.6 --w-second 0.4
```

## Real data format

For real builds, create a manifest like this:

```text
sample_id,layer,specimen_id,class_idx,class_name,ot_path,mpm_path,pbi_path,laser_power_w,scan_speed_mm_s,hatch_distance_mm,layer_thickness_mm,ved_j_mm3
```

Minimum required columns for one channel are:

```text
sample_id,class_idx,class_name,ot_path
```

or:

```text
sample_id,class_idx,class_name,pbi_path
```

The demo labels are:

```text
0 = stable
1 = low_energy_risk
2 = high_energy_risk
```

They can be replaced with labels from CT, metallography, density measurement, surface inspection, or another validation method.

## What should be improved next

The most useful next step is to replace the generated patches with real aligned OT and powder-bed images. After that, the model should be tested with a split that separates builds or specimens, not only random patches. The physical part can also be improved by adding layer-history features, heat accumulation, scan strategy information, and better sensor preprocessing.

## Tests

```bash
pytest -q
```

The current tests check the VED calculation and the late-fusion probability logic.
