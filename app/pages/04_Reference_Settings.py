
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from app.ui_kit import hero, inject_global_styles, section_header
from am_defect_detection.reference_profiles import (
    DEMO_REFERENCE,
    derive_reference_from_manifest,
    make_user_reference_profile,
    normalize_against_reference,
)


st.set_page_config(page_title="Reference Settings", layout="wide")
inject_global_styles()

hero(
    "Reference settings",
    "Define how process parameters are normalized in the app.",
    "A reference VED is not universal. It should be material-, machine-, powder-, and validation-specific whenever possible.",
)

st.warning(
    "The demo reference is only a normalization anchor for the current synthetic workflow. "
    "Do not report it as a universal optimum processing window."
)

section_header(
    "Reference source",
    "Choose whether the app uses the demo reference, a user-defined reference, or a reference estimated from a manifest.",
)

source = st.radio(
    "Reference source",
    ["Demo reference", "User-defined reference", "Derive from manifest"],
    horizontal=True,
)

profile = DEMO_REFERENCE
warnings: list[str] = []

if source == "User-defined reference":
    c1, c2, c3 = st.columns(3)
    ref_ved = c1.number_input("Reference VED [J/mm³]", min_value=1.0, max_value=300.0, value=37.78, step=1.0)
    ref_spot = c2.number_input("Reference spot size / beam diameter [µm]", min_value=10.0, max_value=300.0, value=80.0, step=5.0)
    material = c3.text_input("Material / domain", value="user-defined")
    profile = make_user_reference_profile(ref_ved, ref_spot, material=material)

elif source == "Derive from manifest":
    uploaded = st.file_uploader("Upload manifest CSV", type=["csv"])
    if uploaded is not None:
        tmpdir = Path(tempfile.mkdtemp(prefix="reference_manifest_"))
        manifest_path = tmpdir / uploaded.name
        manifest_path.write_bytes(uploaded.getvalue())

        try:
            profile, warnings = derive_reference_from_manifest(manifest_path)
            st.success("Reference derived from manifest.")
            with st.expander("Manifest preview", expanded=False):
                st.dataframe(pd.read_csv(manifest_path).head(50), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not derive reference: {exc}")
            st.stop()
    else:
        st.info("Upload a manifest to derive a dataset-specific reference.")
        st.stop()

st.subheader("Selected reference")

c1, c2, c3 = st.columns(3)
c1.metric("Reference VED [J/mm³]", f"{profile.reference_ved_j_mm3:.2f}")
c2.metric("Reference spot size [µm]", f"{profile.reference_spot_size_um:.1f}")
c3.metric("Source", profile.source)

st.write("**Description:**", profile.description)
st.write("**Caution:**", profile.caution)

if warnings:
    st.subheader("Reference warnings")
    for w in warnings:
        st.warning(w)

section_header(
    "Parameter check against selected reference",
    "Use this calculator to see how a parameter set compares with the selected reference.",
)

c1, c2, c3, c4, c5 = st.columns(5)
laser_power_w = c1.number_input("Laser power [W]", min_value=1.0, max_value=1000.0, value=200.0, step=10.0)
scan_speed_mm_s = c2.number_input("Scan speed [mm/s]", min_value=1.0, max_value=5000.0, value=800.0, step=50.0)
hatch_distance_mm = c3.number_input("Hatch distance [mm]", min_value=0.01, max_value=0.5, value=0.12, step=0.01)
layer_thickness_mm = c4.number_input("Layer thickness [mm]", min_value=0.005, max_value=0.2, value=0.03, step=0.005)
spot_size_um = c5.number_input("Spot size [µm]", min_value=10.0, max_value=300.0, value=80.0, step=5.0)

norm = normalize_against_reference(
    laser_power_w=laser_power_w,
    scan_speed_mm_s=scan_speed_mm_s,
    hatch_distance_mm=hatch_distance_mm,
    layer_thickness_mm=layer_thickness_mm,
    spot_size_um=spot_size_um,
    reference=profile,
)

st.subheader("Normalized process descriptors")

m1, m2, m3, m4 = st.columns(4)
m1.metric("VED [J/mm³]", f"{norm['ved_j_mm3']:.2f}")
m2.metric("VED / reference", f"{norm['ved_reference_ratio']:.2f}")
m3.metric("Power density [W/mm²]", f"{norm['power_density_w_mm2']:.0f}")
m4.metric("Hatch / spot ratio", f"{norm['hatch_to_spot_ratio']:.2f}")

classification = str(norm["ved_reference_class"])
if classification == "near_reference":
    st.info("The VED is near the selected reference range. This does not guarantee density or absence of defects.")
elif classification == "below_reference":
    st.warning("The VED is below the selected reference. Lack-of-fusion risk may increase, depending on material, overlap, and melt-pool stability.")
else:
    st.warning("The VED is above the selected reference. Keyhole, evaporation, spatter, or overheating risk may increase, depending on beam concentration and material response.")

st.subheader("Why this matters")
st.markdown(
    """
- VED is useful for first-order comparison, but it hides the difference between power, speed, hatch, and layer thickness.
- Spot size changes beam area and therefore power density.
- A reference value should be calibrated to the material, machine, powder condition, optical setup, and ground-truth measurements.
- For reporting, state whether the reference is demo-based, user-defined, or dataset-derived.
"""
)
