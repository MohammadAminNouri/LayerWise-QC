from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Guide", layout="wide")

st.title("Guide")
st.caption("How to use LayerWise-QC without overinterpreting the results.")

st.subheader("Recommended workflow")

st.markdown("""
1. **Dashboard**  
   Use the main dashboard to inspect the process condition, VED, spot size, sensor indicators, fusion result, and feed-forward recommendation.

2. **Data Readiness**  
   Use this before training or reporting accuracy. It checks process parameters, ground truth, sensor paths, split information, and leakage risk.

3. **Validation report**  
   Use the generated reports to decide what can be claimed: workflow demonstration, training experiment, or validation result.
""")

st.subheader("What the app can and cannot claim")

st.markdown("""
| Situation | Supported statement | Not supported |
|---|---|---|
| Synthetic or demo data | The workflow is demonstrated | Real defect detection |
| Literature-derived data | Process trends can be explored | Layer-wise sensor validation |
| Real data without ground truth | Data can be inspected | Model accuracy |
| Real data with ground truth and grouped split | Internal validation can be reported | Generalization to all machines or materials |
| External unseen-build test | Stronger validation within the tested domain | Unlimited generalization |
""")

st.subheader("Key terms")

with st.expander("Process terms"):
    st.markdown("""
- **VED**: volumetric energy density from power, speed, hatch distance, and layer thickness.
- **Spot size / beam diameter**: effective laser diameter on the powder bed.
- **Power density**: laser power divided by beam area.
- **Hatch/spot ratio**: scan-track spacing relative to beam diameter.
""")

with st.expander("Validation terms"):
    st.markdown("""
- **Ground truth**: independent measurement such as CT porosity, density, microscopy, roughness, or mechanical testing.
- **Group leakage**: the same build or specimen appears in both training and testing.
- **Claim level**: the strength of statement supported by the available data.
""")

st.subheader("Minimum useful dataset")

st.code("""sample_id
build_id
specimen_id
layer_id
split
laser_power_w
scan_speed_mm_s
hatch_distance_mm
layer_thickness_mm
spot_size_um
label or class_name or quality_label
ground-truth measurement if available
sensor paths if available""")

st.info("Keep the workflow simple: Dashboard for interpretation, Guide for explanation, and Data Readiness for dataset checking.")
