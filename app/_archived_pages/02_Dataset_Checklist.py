
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui_kit import hero, inject_global_styles, section_header

st.set_page_config(page_title="Dataset Checklist", layout="wide")
inject_global_styles()

hero(
    "Dataset checklist",
    "A practical checklist for preparing LPBF data before training or reporting model accuracy.",
    "Use this page to plan what must be collected from the machine, sensors, and post-build characterization.",
)

section_header(
    "Minimum dataset structure",
    "Define the required columns before collecting or importing data.",
)

minimum = pd.DataFrame(
    [
        ["sample_id", "Required", "Unique row identifier.", "Needed for traceability."],
        ["build_id", "Required for validation", "Identifies the build job.", "Needed for group-wise split."],
        ["specimen_id", "Required for validation", "Identifies the specimen or coupon.", "Prevents specimen leakage."],
        ["layer_id", "Recommended", "Identifies layer number.", "Useful for layer-wise monitoring."],
        ["split", "Required for training", "train, val, or test.", "Defines evaluation protocol."],
        ["material/alloy", "Recommended", "Material family and alloy.", "Defines domain of applicability."],
        ["machine", "Recommended", "LPBF machine or platform.", "Affects transferability."],
    ],
    columns=["Column", "Priority", "Description", "Reason"],
)
st.dataframe(minimum, use_container_width=True, hide_index=True)

section_header(
    "Process parameters",
    "These are needed to compute physics-informed features.",
)

process = pd.DataFrame(
    [
        ["laser_power_w", "Required", "Laser power in watts."],
        ["scan_speed_mm_s", "Required", "Scan speed in mm/s."],
        ["hatch_distance_mm", "Required", "Distance between adjacent scan tracks."],
        ["layer_thickness_mm", "Required", "Layer thickness."],
        ["spot_size_um", "Strongly recommended", "Laser spot size or beam diameter."],
        ["preheat_temperature_c", "Optional", "Build plate or chamber preheat temperature."],
        ["oxygen_ppm", "Optional", "Chamber oxygen level."],
        ["powder_reuse_count", "Optional", "Powder reuse or recycling count."],
        ["scan_strategy", "Optional", "Stripe, chessboard, rotation angle, contour settings."],
    ],
    columns=["Column", "Priority", "Description"],
)
st.dataframe(process, use_container_width=True, hide_index=True)

section_header(
    "Sensor evidence",
    "Use sensor data only when it can be mapped to the same build, specimen, and layer.",
)

sensor = pd.DataFrame(
    [
        ["ot_path", "Optical tomography image path.", "Useful for layer-wise optical anomalies."],
        ["mpm_path", "Melt-pool monitoring image/signal path.", "Useful for melt-pool instability."],
        ["pbi_path", "Powder-bed image path.", "Useful for recoating and powder-layer anomalies."],
        ["pyrometry_path", "Pyrometry or thermal signal path.", "Useful for thermal history."],
        ["machine_log_path", "Machine process log path.", "Useful for alarms, oxygen, recoater, and job events."],
    ],
    columns=["Column", "Description", "Use"],
)
st.dataframe(sensor, use_container_width=True, hide_index=True)

section_header(
    "Ground truth",
    "These measurements turn monitoring data into a supervised validation problem.",
)

truth = pd.DataFrame(
    [
        ["relative_density_pct", "High value", "Archimedes, image analysis, or other density method."],
        ["porosity_pct", "High value", "CT or metallography."],
        ["defect_type", "High value", "Lack of fusion, keyhole, crack, recoating defect, etc."],
        ["surface_roughness_um", "Medium value", "Surface quality target."],
        ["tensile_strength_mpa", "Medium value", "Mechanical performance target."],
        ["elongation_pct", "Medium value", "Ductility target."],
        ["quality_label", "Useful", "Human-readable class, preferably derived from measurements."],
    ],
    columns=["Column", "Priority", "Description"],
)
st.dataframe(truth, use_container_width=True, hide_index=True)

section_header(
    "Recommended collection plan",
    "A realistic plan for moving from demo data to a defensible experiment.",
)

st.markdown(
    """
1. Select one material and one machine first.
2. Print several builds, not only one build.
3. Include repeated specimens for each process condition.
4. Save process parameters and machine logs for every sample.
5. Export aligned sensor images or signals by layer or region.
6. Measure ground truth with CT, density, metallography, roughness, or mechanical tests.
7. Split by build or specimen, not by random rows.
8. Keep an external test build untouched until the final evaluation.
"""
)
