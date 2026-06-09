
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui_kit import hero, inject_global_styles, section_header


st.set_page_config(page_title="Glossary", layout="wide")
inject_global_styles()

hero(
    "Glossary and interpretation guide",
    "Definitions used in LayerWise-QC, written for users who need to understand the workflow before trusting any result.",
    "This page is useful for presentations, supervision meetings, and onboarding new users.",
)

section_header(
    "Core process terms",
    "Explain the process descriptors used by the app.",
    "Use this when a user needs to understand VED, beam diameter, power density, and overlap before reading the dashboard.",
)

process_terms = pd.DataFrame(
    [
        {
            "Term": "VED",
            "Meaning": "Volumetric energy density calculated from laser power, scan speed, hatch distance, and layer thickness.",
            "Why it matters": "It is a useful first process descriptor, but it does not fully describe melt-pool physics.",
            "Common mistake": "Treating similar VED values as equivalent across different machines or beam sizes.",
        },
        {
            "Term": "Spot size / beam diameter",
            "Meaning": "Effective laser diameter on the powder bed.",
            "Why it matters": "It controls how concentrated the energy is.",
            "Common mistake": "Ignoring it when comparing parameter sets.",
        },
        {
            "Term": "Power density",
            "Meaning": "Laser power divided by beam area.",
            "Why it matters": "High power density can increase keyhole or spatter risk even when VED looks acceptable.",
            "Common mistake": "Using power alone without beam area.",
        },
        {
            "Term": "Hatch/spot ratio",
            "Meaning": "Hatch distance divided by beam diameter.",
            "Why it matters": "It indicates scan-track overlap.",
            "Common mistake": "Assuming hatch distance is meaningful without beam diameter.",
        },
        {
            "Term": "Line energy",
            "Meaning": "Laser power divided by scan speed.",
            "Why it matters": "It describes energy delivered along the scan path.",
            "Common mistake": "Using it alone without hatch and layer thickness.",
        },
    ]
)
st.dataframe(process_terms, use_container_width=True, hide_index=True)

section_header(
    "Data and validation terms",
    "Explain why dataset structure is as important as the model.",
)

validation_terms = pd.DataFrame(
    [
        {
            "Term": "Ground truth",
            "Meaning": "Independent measurement such as CT porosity, density, microscopy, roughness, or mechanical test result.",
            "Why it matters": "Without ground truth, the model cannot be validated.",
        },
        {
            "Term": "Group leakage",
            "Meaning": "Rows from the same build or specimen appear in both training and testing.",
            "Why it matters": "It can make accuracy look high while the model is only memorizing build-specific patterns.",
        },
        {
            "Term": "Demo/proxy mode",
            "Meaning": "Transparent rule-based workflow using synthetic data.",
            "Why it matters": "Good for explanation, not for real defect-detection claims.",
        },
        {
            "Term": "Literature-derived benchmark",
            "Meaning": "Manually extracted values from papers or supplementary data.",
            "Why it matters": "Useful for workflow testing and process-property baselines, not a substitute for aligned sensor data.",
        },
        {
            "Term": "Claim level",
            "Meaning": "The strength of statement allowed by the data and validation protocol.",
            "Why it matters": "Prevents overclaiming from synthetic or incomplete data.",
        },
    ]
)
st.dataframe(validation_terms, use_container_width=True, hide_index=True)

section_header(
    "Sensor terms",
    "Clarify what each sensor modality contributes.",
)

sensor_terms = pd.DataFrame(
    [
        {
            "Sensor": "Optical tomography",
            "What it observes": "Layer-wise optical emission or surface-related signatures.",
            "Typical value": "Can indicate overheating, lack of emission, abnormal tracks, or layer anomalies.",
            "Limitation": "Requires alignment to layer/sample and independent ground truth.",
        },
        {
            "Sensor": "Melt-pool monitoring",
            "What it observes": "Melt-pool intensity, geometry, or thermal emission during exposure.",
            "Typical value": "Sensitive to energy input and melt-pool instability.",
            "Limitation": "Raw signals need calibration and synchronization.",
        },
        {
            "Sensor": "Powder-bed imaging",
            "What it observes": "Powder spreading, recoater streaks, exposed layer surface, particles, and local defects.",
            "Typical value": "Useful for recoating and powder-bed anomalies.",
            "Limitation": "Does not directly prove final internal porosity.",
        },
        {
            "Sensor": "Pyrometry",
            "What it observes": "Temperature or thermal radiation signal.",
            "Typical value": "Useful for thermal history and overheating indicators.",
            "Limitation": "Affected by emissivity, optics, and calibration.",
        },
    ]
)
st.dataframe(sensor_terms, use_container_width=True, hide_index=True)

st.info(
    "Use these definitions in reports and presentations so the app is read as a research workflow, not as a black-box defect detector."
)
