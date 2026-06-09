from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from app.ui_kit import inject_global_styles, next_steps_box

from am_defect_detection.data_readiness import (
    audit_dataset_readiness,
    readiness_report_to_markdown,
)

st.set_page_config(page_title="Dataset Readiness", layout="wide")
inject_global_styles()

st.title("Dataset Readiness / Real Data Ingestion")
st.caption("Check whether a manifest is ready for demo, training, or stronger scientific claims.")

st.warning(
    "This page checks data structure, ground truth, sensor availability, and leakage risk. "
    "It does not prove model accuracy."
)

source = st.radio(
    "Choose input source",
    ["Use demo manifest", "Upload manifest CSV"],
    horizontal=True,
)

manifest_path: Path | None = None
root_dir: Path | None = None

if source == "Use demo manifest":
    manifest_path = Path("data/demo_samples/manifest.csv")
    root_dir = Path("data/demo_samples")
    st.info(f"Using demo manifest: `{manifest_path}`")
else:
    uploaded = st.file_uploader("Upload manifest CSV", type=["csv"])
    if uploaded is None:
        st.stop()
    tmpdir = Path(tempfile.mkdtemp(prefix="layerwise_manifest_"))
    manifest_path = tmpdir / uploaded.name
    manifest_path.write_bytes(uploaded.getvalue())
    root_dir = tmpdir
    st.success(f"Uploaded `{uploaded.name}`")

require_images = st.checkbox(
    "Require image files to exist",
    value=False,
    help="Use this only when image paths are accessible in this environment.",
)

try:
    df_preview = pd.read_csv(manifest_path)
except Exception as exc:
    st.error(f"Could not read manifest: {exc}")
    st.stop()

with st.expander("Manifest preview", expanded=True):
    st.dataframe(df_preview.head(50), use_container_width=True)
    st.caption(f"Rows: {len(df_preview)} | Columns: {len(df_preview.columns)}")

report = audit_dataset_readiness(
    manifest_path=manifest_path,
    root_dir=root_dir,
    require_images=require_images,
)

st.subheader("Readiness summary")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", report.n_rows)
c2.metric("Errors", report.n_errors)
c3.metric("Warnings", report.n_warnings)
c4.metric("OK for training", "Yes" if report.ok_for_training else "No")
c5.metric("OK for strong claims", "Yes" if report.ok_for_claims else "No")

if report.ok_for_claims:
    st.success("Dataset structure looks strong enough for serious validation experiments.")
elif report.ok_for_training:
    st.info("Dataset may be usable for training experiments, but claims are limited.")
elif report.ok_for_demo:
    st.warning("Dataset may be usable for workflow testing only.")
else:
    st.error("Dataset is not ready. Fix errors first.")

st.subheader("Available information")

a, b, c, d = st.columns(4)
a.write("**Sensor modalities**")
a.write(report.available_modalities or "None")
b.write("**Ground truth columns**")
b.write(report.available_ground_truth or "None")
c.write("**Process columns**")
c.write(report.process_columns_present or "None")
d.write("**Group columns**")
d.write(report.group_columns_present or "None")

if report.class_counts:
    st.subheader("Class counts")
    st.bar_chart(pd.Series(report.class_counts))

if report.split_counts:
    st.subheader("Split counts")
    st.bar_chart(pd.Series(report.split_counts))

if report.parameter_ranges:
    st.subheader("Parameter ranges")
    st.dataframe(pd.DataFrame(report.parameter_ranges).T, use_container_width=True)

st.subheader("Issues and recommendations")
if report.issues:
    st.dataframe(pd.DataFrame([i.__dict__ for i in report.issues]), use_container_width=True)
else:
    st.success("No issues found.")

md = readiness_report_to_markdown(report)

st.subheader("Download reports")
st.download_button(
    "Download Markdown report",
    data=md,
    file_name="dataset_readiness_report.md",
    mime="text/markdown",
)

st.download_button(
    "Download JSON report",
    data=json.dumps(report.to_dict(), indent=2),
    file_name="dataset_readiness_report.json",
    mime="application/json",
)

with st.expander("Markdown report preview"):
    st.markdown(md)
    

st.subheader("Recommended next actions")

steps = []
codes = {issue.code for issue in report.issues}

if "missing_build_id" in codes:
    steps.append("Add build_id so training and testing can be separated by build.")
if "missing_specimen_id" in codes:
    steps.append("Add specimen_id so samples from the same specimen do not appear in both train and test.")
if "missing_split" in codes:
    steps.append("Add a split column using train, val, and test. Prefer grouped splitting by build_id.")
if "no_ground_truth" in codes:
    steps.append("Add independent ground truth such as porosity, density, microscopy result, roughness, or defect type.")
if "no_sensor_modalities" in codes:
    steps.append("Add sensor paths when available, such as OT, MPM, PBI, pyrometry, or machine logs.")
if "missing_spot_size_um" in codes:
    steps.append("Add spot_size_um or beam diameter when available, because VED alone does not capture beam concentration.")
if not steps:
    steps.append("Proceed to feature-table generation and grouped validation.")

next_steps_box(steps)
