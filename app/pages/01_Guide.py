
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Guide", layout="wide")

st.title("Guide")
st.caption("A compact guide for using LayerWise-QC without overinterpreting the results.")

st.info(
    "The app has three main areas: Dashboard for interpretation, Data Readiness for dataset checks, "
    "and this Guide for definitions, workflow, dataset requirements, references, and validation rules."
)

tabs = st.tabs(
    [
        "Workflow",
        "Terms",
        "Dataset checklist",
        "Reference settings",
        "Validation protocol",
        "Dashboard tabs",
    ]
)

with tabs[0]:
    st.subheader("Recommended workflow")

    st.markdown(
        """
1. **Dashboard**  
   Inspect the process condition, VED, spot size, sensor indicators, fusion result, and feed-forward recommendation.

2. **Data Readiness**  
   Check whether the manifest has process parameters, ground truth, sensor paths, split information, and leakage control.

3. **Feature table and baselines**  
   Build the feature table and compare process-only, sensor-only, and hybrid baselines.

4. **Validation report**  
   Report what the current evidence supports: workflow demonstration, training experiment, internal validation, or external validation.
"""
    )

    st.subheader("Claim level")

    st.markdown(
        """
| Situation | Supported statement | Not supported |
|---|---|---|
| Synthetic or demo data | The workflow is demonstrated | Real defect detection |
| Literature-derived data | Process trends can be explored | Layer-wise sensor validation |
| Real data without ground truth | Data can be inspected | Model accuracy |
| Real data with ground truth and grouped split | Internal validation can be reported | Generalization to all machines or materials |
| External unseen-build test | Stronger validation within the tested domain | Unlimited generalization |
"""
    )

with tabs[1]:
    st.subheader("Process terms")

    st.markdown(
        """
| Term | Meaning | Why it matters |
|---|---|---|
| VED | Volumetric energy density from power, speed, hatch distance, and layer thickness | Useful first descriptor, but not a universal predictor |
| Spot size / beam diameter | Effective laser diameter on the powder bed | Controls beam area and energy concentration |
| Power density | Laser power divided by beam area | Helps interpret keyhole/spatter tendency |
| Hatch/spot ratio | Hatch distance relative to beam diameter | Helps interpret scan-track overlap |
| Line energy | Laser power divided by scan speed | Describes energy along the scan path |
"""
    )

    st.subheader("Validation terms")

    st.markdown(
        """
| Term | Meaning | Why it matters |
|---|---|---|
| Ground truth | Independent measurement such as CT porosity, density, microscopy, roughness, or mechanical testing | Needed for model validation |
| Group leakage | Same build or specimen appears in both training and testing | Can inflate accuracy |
| Literature-derived benchmark | Manually extracted paper data | Useful for workflow testing, not sensor validation |
| Claim level | Strength of statement supported by the data | Prevents overclaiming |
"""
    )

    st.subheader("Sensor terms")

    st.markdown(
        """
| Sensor | What it can show | Limitation |
|---|---|---|
| Optical tomography | Layer-wise optical anomalies | Needs alignment and ground truth |
| Melt-pool monitoring | Melt-pool intensity or instability | Needs synchronization and calibration |
| Powder-bed imaging | Recoating streaks, powder-bed defects, surface anomalies | Does not directly prove internal porosity |
| Pyrometry | Thermal signal or temperature proxy | Sensitive to emissivity and calibration |
| Machine logs | Alarms, oxygen, recoater, job events | Often indirect and machine-specific |
"""
    )

with tabs[2]:
    st.subheader("Minimum useful manifest")

    st.code(
        """sample_id
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
sensor paths if available""",
        language="text",
    )

    st.subheader("Process parameters")

    st.markdown(
        """
| Column | Priority | Description |
|---|---|---|
| laser_power_w | Required | Laser power in watts |
| scan_speed_mm_s | Required | Scan speed in mm/s |
| hatch_distance_mm | Required | Distance between scan tracks |
| layer_thickness_mm | Required | Layer thickness |
| spot_size_um | Strongly recommended | Laser spot size or beam diameter |
| preheat_temperature_c | Optional | Build plate or chamber preheat |
| oxygen_ppm | Optional | Chamber oxygen level |
| powder_reuse_count | Optional | Powder reuse state |
| scan_strategy | Optional | Stripe, chessboard, rotation angle, contour strategy |
"""
    )

    st.subheader("Ground truth")

    st.markdown(
        """
| Column | Value |
|---|---|
| relative_density_pct | Density target |
| porosity_pct | Porosity target from CT or metallography |
| defect_type | Lack of fusion, keyhole, crack, recoating defect, etc. |
| surface_roughness_um | Surface quality target |
| tensile_strength_mpa | Mechanical target |
| elongation_pct | Ductility target |
"""
    )

    st.subheader("Collection plan")

    st.markdown(
        """
1. Start with one material and one machine.
2. Print several builds, not only one build.
3. Include repeated specimens for each process condition.
4. Save process parameters and machine logs for every sample.
5. Export aligned sensor images or signals by layer or region.
6. Measure ground truth with CT, density, metallography, roughness, or mechanical testing.
7. Split by build or specimen, not by random rows.
8. Keep one external test build untouched until the final evaluation.
"""
    )

with tabs[3]:
    st.subheader("Reference values")

    st.markdown(
        """
The app should not treat one VED value as universal. A reference value is only a normalization anchor.

Preferred order:

1. **Dataset-derived reference** from validated acceptable parts.
2. **User-defined reference** from the machine/material/process window.
3. **Demo reference** only for synthetic workflow explanation.
"""
    )

    st.subheader("Why reference settings matter")

    st.markdown(
        """
- VED hides different combinations of power, speed, hatch distance, and layer thickness.
- Spot size changes beam area and power density.
- A parameter set can have similar VED but different melt-pool behavior.
- Reference values should be reported together with material, machine, powder state, optical setup, and validation method.
"""
    )

    st.subheader("What to report")

    st.code(
        """reference_source: demo / user_defined / dataset_derived
reference_ved_j_mm3
reference_spot_size_um
material
machine
powder_state
ground_truth_method
validation_split_method""",
        language="text",
    )

with tabs[4]:
    st.subheader("Validation levels")

    st.markdown(
        """
| Level | Data | Supported statement |
|---|---|---|
| Workflow demonstration | Synthetic or toy data | The software pipeline works |
| Process-property baseline | Literature-derived or small experimental table | Process trends can be explored |
| Internal validation | Real aligned data, split by build/specimen | Predictive value within the tested domain |
| External validation | Independent unseen builds | Stronger evidence of generalization within domain |
"""
    )

    st.subheader("Recommended metrics")

    st.markdown(
        """
| Metric | Use |
|---|---|
| Balanced accuracy | Classification with imbalance |
| MCC | Single robust classification score |
| Macro F1 | Treats each class equally |
| Confusion matrix | Shows failure modes |
| MAE/RMSE | Regression targets such as porosity or density |
| Calibration curve | Checks whether confidence scores are reliable |
| Per-build performance | Shows whether one build dominates the result |
"""
    )

    st.subheader("Splitting rules")

    st.markdown(
        """
Best practice:

1. Leave-one-build-out validation.
2. Group split by build_id.
3. Group split by specimen_id.
4. Random row split only for debugging.

Avoid final accuracy claims from random row splits when several layers or patches come from the same build.
"""
    )

with tabs[5]:
    st.subheader("Main dashboard tabs")

    st.markdown(
        """
| Tab | Purpose | Best next action |
|---|---|---|
| Overview | Summarize current state, risk direction, and claim level | Check whether result is demo or validation-based |
| Process inputs | Define process parameters and reference values | Confirm material, machine, spot size, and reference |
| Live decision | Show current risk direction and reasons | Read the top reasons and check evidence |
| Sensor signals | Display sensor evidence | Check availability, alignment, and limitations |
| Physics features | Explain process descriptors | Compare with selected reference |
| Sensor fusion | Combine process and sensor evidence | Check modality disagreement |
| Feed-forward control | Suggest conservative process adjustment | Validate experimentally before applying |
| Data / manifest | Check dataset structure | Fix missing ground truth, split, and group IDs |
"""
    )

    st.warning(
        "A dashboard tab should always show purpose, main output, limitation, and next action. "
        "This keeps the interface useful during meetings and avoids overclaiming."
    )
