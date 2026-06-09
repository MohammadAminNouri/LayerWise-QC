# LayerWise-QC

LayerWise-QC is a research prototype for layer-wise quality monitoring in laser powder bed fusion additive manufacturing.

The project connects LPBF process parameters, physics-based descriptors, sensor-derived indicators, dataset validation, machine-learning baselines, model-evaluation utilities, sensor-fusion logic, and validation reporting. It is designed to support a clear research workflow from process condition to evidence, model output, uncertainty, and claim level.

The project should not be presented as a validated industrial defect detector unless it is used with real aligned sensor data, independent ground truth, and leakage-safe validation.

---

## 1. Purpose

LayerWise-QC is built to answer four questions:

1. Are the LPBF process parameters physically reasonable?
2. Is the dataset structured well enough for training or validation?
3. Which process, physics, and sensor features are associated with quality risk?
4. What level of claim is supported by the available evidence?

The software separates workflow demonstration from model validation. Synthetic or literature-derived data can test the pipeline, but real accuracy claims require experimental data and independent measurements.

---

## 2. Main workflow

The intended workflow is:

```text
process parameters
+ spot size / beam diameter
+ sensor images or sensor descriptors
+ machine/log information
+ ground-truth measurements
-> dataset audit
-> feature-table generation
-> tabular baselines
-> image models where available
-> sensor fusion and uncertainty
-> validation report
```

The app supports three practical levels of use:

| Level | Data source | Supported statement |
|---|---|---|
| Demo/proxy mode | Synthetic examples and transparent rules | The workflow can be demonstrated |
| Literature-derived benchmark | Manually extracted paper data or open supplementary tables | Process-property trends can be explored |
| Real validation mode | Aligned sensor data with independent ground truth | Internal or external validation can be reported within the tested domain |

---

## 3. App structure

The Streamlit app is intentionally kept simple in the sidebar.

### Dashboard

The main dashboard shows the current LPBF process condition and its interpretation. It includes laser power, scan speed, hatch distance, layer thickness, spot size, VED, beam area, power density, hatch/spot ratio, sensor indicators, fusion result, uncertainty, and feed-forward advisory recommendations.

In demo mode, the dashboard uses transparent proxy logic. This makes the reasoning visible, but it should not be reported as a validated defect prediction.

### Guide

The Guide page contains the explanation material that was previously split across many pages. It covers workflow, key terms, dataset checklist, reference settings, validation protocol, and dashboard-tab interpretation.

### Data Readiness

The Data Readiness page checks whether a manifest is suitable for workflow testing, training, or stronger validation. It checks required process columns, spot size, sensor modalities, ground truth, class balance, split quality, leakage risk, missing values, and claim level.

---

## 4. Process physics

The first process descriptor is volumetric energy density:

```text
VED = laser_power_w / (scan_speed_mm_s * hatch_distance_mm * layer_thickness_mm)
```

VED is useful, but it is not sufficient by itself. Similar VED values can produce different melt-pool behavior when power, speed, hatch distance, layer thickness, beam diameter, material, absorptivity, or scan strategy differ.

For that reason, the project also includes spot-size and beam-derived descriptors:

```text
spot_size_um
spot_size_mm
beam_radius_mm
beam_area_mm2
power_density_w_mm2
normalized_spot_size
spot_overlap_ratio
hatch_to_spot_ratio
spot_to_layer_ratio
power_density_x_residence_proxy
ved_x_power_density
```

This separates energy per volume from energy concentration on the powder bed.

---

## 5. Reference values

A reference VED should not be treated as a universal optimum. It is a normalization anchor.

| Reference type | Use |
|---|---|
| Demo reference | Used only for synthetic workflow explanation |
| User-defined reference | Entered from a known machine/material parameter window |
| Dataset-derived reference | Estimated from acceptable or standard rows in a manifest |

A reference should be reported together with material, machine, powder condition, spot size, ground-truth method, and validation split method.

---

## 6. Data manifest

A useful manifest should contain at least:

```text
sample_id
build_id
specimen_id
layer_id
split
material
machine
laser_power_w
scan_speed_mm_s
hatch_distance_mm
layer_thickness_mm
spot_size_um
label or class_name or quality_label
ground-truth measurement if available
sensor paths if available
```

Recommended sensor columns:

```text
ot_path
mpm_path
pbi_path
pyrometry_path
machine_log_path
```

Recommended ground-truth columns:

```text
relative_density_pct
porosity_pct
surface_roughness_um
defect_type
quality_label
tensile_strength_mpa
elongation_pct
```

For validation, the most important fields are `build_id`, `specimen_id`, `split`, and a ground-truth measurement. Without these fields, reported model accuracy can be misleading.

---

## 7. Dataset readiness

The dataset-readiness audit checks whether a manifest can support workflow testing, training experiments, or stronger validation.

It checks missing process parameters, invalid numeric values, missing spot size, missing ground truth, missing sensor paths, duplicated sample IDs, class imbalance, missing split columns, invalid split names, build/specimen leakage, and missing build/specimen identifiers.

The audit does not prove model accuracy. It only checks whether the dataset is structured well enough for the next step.

---

## 8. Machine-learning components

LayerWise-QC includes several machine-learning components. Each has a different role.

### 8.1 Tabular baselines

The tabular pipeline trains models using process parameters, physics features, and sensor descriptors.

| Input group | Purpose |
|---|---|
| Process-only | Tests what can be learned from laser parameters and physics descriptors |
| Sensor-only | Tests what can be learned from image/signal descriptors |
| Process + sensor hybrid | Tests whether sensor evidence improves over process parameters alone |

Typical models include logistic regression, random forest, gradient boosting, or histogram gradient boosting.

Typical outputs:

```text
metrics.json
predictions.csv
feature_importance.csv
leaderboard.csv
```

Metrics include balanced accuracy, MCC, macro F1, weighted F1, and confusion matrix.

### 8.2 Image models

The image pipeline supports modality-specific training for sensor images such as optical tomography, melt-pool monitoring, or powder-bed imaging.

The image-model workflow includes manifest-based image loading, train/validation/test split, class weighting or weighted sampling, CNN or ResNet-style model training, prediction export, and Grad-CAM support.

Image models are meaningful only when the sensor images are aligned with the correct build, specimen, layer, process condition, and ground-truth measurement.

### 8.3 Late fusion

The fusion logic combines predictions or scores from multiple sources, for example process proxy, optical tomography model, melt-pool monitoring model, and powder-bed imaging model.

Fusion can be useful when different modalities capture different failure modes. It also exposes disagreement. If one modality indicates high-energy risk and another indicates standard behavior, uncertainty should increase.

Fusion is only reliable when the individual modalities are calibrated and validated.

### 8.4 Uncertainty and claim level

The app distinguishes between workflow demonstration, training experiment, internal validation, and external validation.

A result based only on synthetic data or proxy rules is not a defect-detection claim. A result based on real aligned data with independent ground truth and grouped validation can support stronger statements, but only inside the tested material, machine, and parameter domain.

---

## 9. Literature-derived benchmark

The repository supports a literature-derived benchmark workflow. This is useful when real in-situ data is not yet available.

Allowed sources include manually entered values, user-provided CSV files, open-access papers, supplementary datasets with clear citation, and user-provided PDFs when legally available.

Every row should preserve citation information. Literature-derived data can support process-property exploration. It cannot replace aligned layer-wise sensor data.

---

## 10. Validation protocol

Final accuracy should not be reported from random row splits when several rows come from the same build or specimen.

Preferred validation order:

1. leave-one-build-out validation
2. group split by `build_id`
3. group split by `specimen_id`
4. random split only for debugging

A validation report should include dataset summary, class distribution, split distribution, group leakage check, feature set, model type, metrics, confusion matrix, per-class performance, per-build performance if available, known limitations, and claim level.

---

## 11. Feed-forward control

The feed-forward module gives conservative advisory recommendations. It may suggest increasing or decreasing energy input depending on the risk direction.

| Risk direction | Possible advisory direction |
|---|---|
| Low-energy / lack-of-fusion tendency | Slightly increase power or reduce scan speed |
| High-energy / keyhole-spatter tendency | Slightly reduce power or increase scan speed |
| Powder-bed or recoating issue | Inspect powder/recoating before changing laser parameters |

These recommendations are advisory only. They should not be treated as automatic machine control.

---

## 12. Installation

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run tests:

```bash
python -m pytest -q
```

Run the app:

```bash
python -m streamlit run app/streamlit_app.py
```

In Codespaces:

```bash
python -m streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

---

## 13. Main commands

Validate a manifest:

```bash
python scripts/validate_manifest.py --manifest data/demo_samples/manifest.csv --root data/demo_samples
```

Audit dataset readiness:

```bash
python scripts/audit_dataset.py \
  --manifest data/demo_samples/manifest.csv \
  --root data/demo_samples \
  --out-md outputs/dataset_readiness_report.md \
  --out-json outputs/dataset_readiness_report.json
```

Build a feature table:

```bash
python scripts/build_feature_table.py \
  --manifest data/demo_samples/manifest.csv \
  --root data/demo_samples \
  --out outputs/features.csv
```

Build literature-derived features:

```bash
python scripts/build_literature_dataset.py \
  --input data/literature/raw_literature_records_template.csv \
  --out outputs/literature_features.csv \
  --manifest-out outputs/literature_manifest.csv
```

Train tabular baselines:

```bash
python scripts/train_tabular_baselines.py \
  --features outputs/features.csv \
  --out outputs/tabular \
  --target class_name \
  --task classification
```

Generate validation report:

```bash
python scripts/generate_validation_report.py \
  --metrics outputs/tabular/metrics.json \
  --predictions outputs/tabular/predictions.csv \
  --manifest data/demo_samples/manifest.csv \
  --data-kind synthetic \
  --out outputs/validation_report.md
```

---

## 14. Expected outputs

Common generated outputs include:

```text
outputs/features.csv
outputs/literature_features.csv
outputs/literature_manifest.csv
outputs/tabular/metrics.json
outputs/tabular/predictions.csv
outputs/tabular/feature_importance.csv
outputs/tabular/leaderboard.csv
outputs/validation_report.md
outputs/dataset_readiness_report.md
outputs/dataset_readiness_report.json
```

Large generated outputs should generally not be committed unless they are small demonstration files.

---

## 15. Current limitations

The current demo data is synthetic or limited. It is useful for testing the workflow, but not for claiming real defect detection.

The main missing element for real accuracy is aligned experimental data:

```text
real sensor images or signals
machine logs
build/specimen/layer identifiers
independent ground truth
group-wise train/test split
external test build
```

Until these are available, the project should be described as a research workflow prototype.

---

## 16. Recommended next experimental step

The next useful experimental step is to prepare a manifest where each row maps build, specimen, layer or region, process parameters, spot size, sensor file paths, ground-truth measurement, and split.

A small but well-structured dataset is more valuable than many unaligned images.

A strong first target is a process + sensor + ground-truth table for one material and one machine, validated with a group-wise split by build or specimen.

---

## 17. Project positioning

LayerWise-QC provides a structured environment for LPBF quality-monitoring research. It connects process physics, sensor descriptors, machine-learning baselines, fusion, data-readiness checks, and validation reporting.

The main value is not only prediction. The main value is that the evidence chain is visible:

```text
process condition
-> physics features
-> sensor evidence
-> model output
-> uncertainty
-> validation status
-> supported claim
```

This structure helps keep the work technically clear and prevents overclaiming before real validation data is available.
