
from __future__ import annotations

import streamlit as st

from app.ui_kit import card, claim_level_box, hero, inject_global_styles, step


st.set_page_config(page_title="Workflow Guide", layout="wide")
inject_global_styles()

hero(
    "LayerWise-QC workflow guide",
    "A structured workflow for LPBF quality monitoring: process parameters, sensor evidence, dataset checks, model training, and validation reporting.",
    "The current demo explains the workflow. Strong defect-detection claims require aligned sensor data and independent ground truth.",
)

st.subheader("Choose the correct mode")

c1, c2, c3 = st.columns(3)

with c1:
    card(
        "Demo dashboard",
        "Use synthetic examples and transparent proxy logic to understand VED, spot size, sensor fusion, uncertainty, and feed-forward recommendations.",
        "Workflow demonstration",
        "demo",
    )

with c2:
    card(
        "Dataset readiness",
        "Check whether a manifest has process parameters, sensor paths, ground truth, split information, and group identifiers for leakage control.",
        "Data audit",
        "training",
    )

with c3:
    card(
        "Validation workflow",
        "Build feature tables, train baselines, compare results, and generate a report that states what can and cannot be claimed.",
        "Research validation",
        "validation",
    )

st.divider()

st.subheader("Recommended order of use")

step(
    "1. Start with the demo dashboard",
    "Use the built-in examples to understand how process conditions affect risk indicators. Treat this as an explanation layer, not a validated detector.",
)

step(
    "2. Audit the dataset",
    "Open Dataset Readiness and check whether the manifest has sample IDs, build IDs, specimen IDs, process parameters, spot size, sensor paths, and ground truth.",
)

step(
    "3. Build the feature table",
    "Convert the manifest into physics features, sensor descriptors, labels, and split information. This is the table used for training and baseline comparisons.",
)

step(
    "4. Train baseline models",
    "Compare process-only, sensor-only, and hybrid models. Use grouped splits so samples from the same build or specimen do not leak into the test set.",
)

step(
    "5. Generate a validation report",
    "Export the dataset summary, metrics, limitations, and claim level. Use this report when discussing the work with supervisors or collaborators.",
)

st.divider()

st.subheader("What the app can claim")

c1, c2 = st.columns(2)

with c1:
    claim_level_box(
        "Current demo level",
        "The app demonstrates a complete LPBF quality-monitoring workflow using synthetic or limited data.",
        "Real aligned sensor data, ground truth, grouped validation, and independent test builds.",
    )

with c2:
    claim_level_box(
        "Target validation level",
        "The model predicts quality indicators on unseen builds within a defined material, machine, and parameter domain.",
        "Sufficient samples per class, reliable labels from CT/density/metallography, and calibration checks.",
    )

st.divider()

st.subheader("Minimum data needed for a serious experiment")

st.markdown(
    """
| Category | Required information | Why it matters |
|---|---|---|
| Identity | `sample_id`, `build_id`, `specimen_id`, `layer_id` | Prevents leakage and preserves traceability |
| Process | power, speed, hatch, layer thickness, spot size | Enables VED, power density, and physics features |
| Sensor evidence | OT, MPM, PBI, pyrometry, or machine logs | Provides in-situ evidence beyond process parameters |
| Ground truth | CT porosity, density, metallography, roughness, defect class | Converts monitoring into supervised validation |
| Split | train, validation, test by build or specimen | Tests generalization instead of memorization |
"""
)

st.divider()

st.subheader("Terms used in the app")

with st.expander("Process descriptors"):
    st.markdown(
        """
- **VED**: volumetric energy density from power, speed, hatch distance, and layer thickness.
- **Spot size / beam diameter**: the effective laser diameter on the powder bed.
- **Power density**: laser power divided by beam area.
- **Hatch/spot ratio**: relation between scan spacing and beam diameter. It helps interpret overlap.
"""
    )

with st.expander("Validation terms"):
    st.markdown(
        """
- **Ground truth**: an independent measurement such as CT porosity, Archimedes density, microscopy, or tensile results.
- **Group leakage**: the same build or specimen appears in both training and testing. This can inflate accuracy.
- **Literature-derived benchmark**: manually extracted paper data used for workflow testing, not a substitute for aligned sensor validation.
- **Claim level**: the strength of statement supported by the available data and validation protocol.
"""
    )

st.divider()

st.subheader("Practical next step")

st.info(
    "Open Dataset Readiness, load the demo manifest, and inspect the warnings. "
    "Those warnings show exactly what must be added before the project can support stronger validation claims."
)
