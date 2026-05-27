# LayerWise-QC Training Patch

This patch adds the first serious training pipeline.

## New files

```text
src/am_defect_detection/feature_table.py
scripts/build_feature_table.py
scripts/train_tabular_baselines.py
scripts/make_better_synthetic_data.py
```

## Why this matters

Your current app is a guided prototype. Training needs a proper feature table and baselines:

1. process-only model
2. sensor-descriptor-only model
3. hybrid process + sensor model

This matches the research direction suggested by your professor: compare data-driven, sensor-driven, and physics-informed modelling.

## Step 1 — create larger synthetic data for testing

```bash
python scripts/make_better_synthetic_data.py \
  --out data/synthetic_v2 \
  --builds 6 \
  --specimens-per-build 12 \
  --layers 25 \
  --regions-per-layer 2
```

This creates:

```text
data/synthetic_v2/manifest.csv
data/synthetic_v2/images/
```

## Step 2 — build the feature table

```bash
python scripts/build_feature_table.py \
  --manifest data/synthetic_v2/manifest.csv \
  --image-root data/synthetic_v2 \
  --out outputs/features/synthetic_v2_features.csv
```

## Step 3 — train classification baselines

```bash
python scripts/train_tabular_baselines.py \
  --features outputs/features/synthetic_v2_features.csv \
  --target class_name \
  --task classification \
  --group-col specimen_id \
  --out-dir outputs/training/synthetic_v2_classification
```

## Step 4 — train regression baselines

```bash
python scripts/train_tabular_baselines.py \
  --features outputs/features/synthetic_v2_features.csv \
  --target relative_density \
  --task regression \
  --group-col specimen_id \
  --out-dir outputs/training/synthetic_v2_density
```

## Outputs

Each training run writes:

```text
metrics.json
leaderboard.csv
predictions.csv
*.joblib trained models
```

## Important

Synthetic data is only for testing code. Real research needs real OT / MPM / PBI data and ground truth such as CT porosity, relative density, or metallography.
