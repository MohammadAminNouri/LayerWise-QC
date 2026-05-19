"""Streamlit dashboard for the AM in-situ monitoring prototype."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from am_defect_detection.constants import CLASS_NAMES, PATCH_SIZES_HW, PROCESS_CONDITIONS  # noqa: E402
from am_defect_detection.simulation import (  # noqa: E402
    ProcessInputs,
    classify_from_ved,
    fuse_scores,
    image_from_process,
    soft_process_scores,
)


st.set_page_config(page_title="In-situ AM quality monitor", layout="wide")

LABELS = {
    "standard": "standard window",
    "delta_minus_30_ved": "low-energy / lack-of-fusion risk",
    "delta_plus_30_ved": "high-energy / keyhole-spatter risk",
}


def _bar_df(scores: dict[str, float], source: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "class": [LABELS[k] for k in CLASS_NAMES],
            "score": [scores[k] for k in CLASS_NAMES],
            "source": source,
        }
    )


def _read_manifest() -> pd.DataFrame | None:
    path = ROOT / "data" / "demo_samples" / "manifest.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _resolve_demo_path(rel: str) -> Path:
    return ROOT / "data" / "demo_samples" / rel


def _show_demo_samples() -> None:
    st.subheader("Built-in sample patches")
    manifest = _read_manifest()
    if manifest is None or manifest.empty:
        st.info("No demo sample manifest found. Run `python scripts/make_synthetic_dataset.py --out data/demo_samples --layers 18`.")
        return

    class_pick = st.selectbox(
        "Pick a sample condition",
        CLASS_NAMES,
        format_func=lambda x: LABELS[x],
        key="sample_condition",
    )
    subset = manifest[manifest["class_name"] == class_pick].reset_index(drop=True)
    row = subset.iloc[0] if len(subset) else manifest.iloc[0]
    cols = st.columns(3)
    for col, modality in zip(cols, ["ot", "mpm", "pbi"]):
        path_col = f"{modality}_path"
        with col:
            st.caption(modality.upper())
            if path_col in row and isinstance(row[path_col], str):
                path = _resolve_demo_path(row[path_col])
                if path.exists():
                    st.image(Image.open(path), use_container_width=True)
                else:
                    st.warning(f"Missing {path.name}")
            else:
                st.warning("not in manifest")

    st.write(
        "The shipped patches are synthetic placeholders. They are only for opening the interface, "
        "checking the data flow, and explaining what each sensor channel contributes."
    )


def _show_parameter_console() -> None:
    st.subheader("Parameter console")
    left, right = st.columns([0.42, 0.58], gap="large")

    with left:
        preset = st.selectbox(
            "Start from condition",
            ["standard", "delta_minus_30_ved", "delta_plus_30_ved"],
            format_func=lambda x: LABELS[x],
        )
        p0 = PROCESS_CONDITIONS[preset]
        laser_power = st.slider("laser power P [W]", 180, 430, int(p0.laser_power_w), 1)
        scan_speed = st.slider("scan speed v [mm/s]", 650, 1700, int(p0.scan_speed_mm_s), 1)
        hatch = st.slider("hatch distance h [mm]", 0.07, 0.18, 0.12, 0.005)
        layer = st.slider("layer thickness L [mm]", 0.02, 0.09, 0.06, 0.005)
        heat_memory = st.slider("heat memory / accumulation", 0.0, 1.0, 0.35, 0.01)
        powder_uniformity = st.slider("powder-bed uniformity", 0.0, 1.0, 0.82, 0.01)
        w_ot = st.slider("OT fusion weight", 0.0, 1.0, 0.50, 0.05)
        w_second = 1.0 - w_ot
        second_modality = st.radio("second channel", ["mpm", "pbi"], horizontal=True)

    inputs = ProcessInputs(
        laser_power_w=laser_power,
        scan_speed_mm_s=scan_speed,
        hatch_distance_mm=hatch,
        layer_thickness_mm=layer,
        heat_memory=heat_memory,
        powder_uniformity=powder_uniformity,
    )
    label = classify_from_ved(inputs.ved)

    # The demo imitates the paper logic: OT tends to be more global, MPM more local;
    # PBI is sensitive to powder uniformity. These small offsets make the complementarity visible.
    ot_inputs = ProcessInputs(laser_power, scan_speed, hatch, layer, heat_memory + 0.05, powder_uniformity)
    second_inputs = ProcessInputs(laser_power, scan_speed, hatch, layer, heat_memory, powder_uniformity - (0.08 if second_modality == "pbi" else 0.0))
    ot_scores = soft_process_scores(ot_inputs)
    second_scores = soft_process_scores(second_inputs)
    fused = fuse_scores(ot_scores, second_scores, w_a=w_ot, w_b=w_second)
    pred = max(fused, key=fused.get)

    with right:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("VED", f"{inputs.ved:.2f} J/mm³")
        m2.metric("relative to 37.78", f"{inputs.ved / 37.78:.2f}×")
        m3.metric("rule window", LABELS[label])
        m4.metric("fused output", LABELS[pred])

        imgs = st.columns(3)
        seed = int(laser_power + scan_speed + 1000 * hatch + 1000 * layer)
        with imgs[0]:
            st.caption("OT simulated patch")
            st.image(image_from_process(inputs, PATCH_SIZES_HW["ot"], "ot", seed=seed), use_container_width=True)
        with imgs[1]:
            st.caption(f"{second_modality.upper()} simulated patch")
            st.image(image_from_process(inputs, PATCH_SIZES_HW[second_modality], second_modality, seed=seed + 3), use_container_width=True)
        with imgs[2]:
            st.caption("Fused class scores")
            chart = pd.concat([_bar_df(fused, "fusion")], ignore_index=True)
            st.bar_chart(chart, x="class", y="score")

        score_table = pd.concat(
            [_bar_df(ot_scores, "ot"), _bar_df(second_scores, second_modality), _bar_df(fused, "fusion")],
            ignore_index=True,
        )
        st.dataframe(score_table.pivot(index="class", columns="source", values="score").round(3), use_container_width=True)


def _show_layer_story() -> None:
    st.subheader("Layer-by-layer sketch")
    n_layers = st.slider("number of intentionally disturbed layers", 1, 12, 7)
    start = st.slider("disturbance starts at layer", 5, 70, 20)
    mode = st.radio("disturbance", ["delta_minus_30_ved", "delta_plus_30_ved"], format_func=lambda x: LABELS[x], horizontal=True)

    xs = np.arange(0, 90)
    risk = np.zeros_like(xs, dtype=float) + 0.08
    risk[(xs >= start) & (xs < start + n_layers)] = 0.72 if mode == "delta_minus_30_ved" else 0.58
    after = xs >= start + n_layers
    risk[after] *= np.exp(-0.22 * (xs[after] - (start + n_layers)))
    # Simple visible reminder of the reported 7-layer healing observation.
    residual_flag = n_layers > 7
    df = pd.DataFrame({"layer": xs, "defect-like signal": risk})
    st.line_chart(df, x="layer", y="defect-like signal")
    if residual_flag:
        st.warning("The disturbed stack is deeper than seven layers, so the sketch keeps some residual risk.")
    else:
        st.success("The disturbed stack is seven layers or less, so the sketch lets the signal decay after standard exposure.")


def main() -> None:
    st.title("In-situ quality monitor for powder-bed fusion")
    st.write(
        "A small working interface around the OT + second-sensor pipeline. "
        "It is meant for discussion: adjust process parameters, inspect example patches, "
        "and see how a late-fusion quality score would move."
    )
    tab1, tab2, tab3 = st.tabs(["live console", "sample images", "layer story"])
    with tab1:
        _show_parameter_console()
    with tab2:
        _show_demo_samples()
    with tab3:
        _show_layer_story()


if __name__ == "__main__":
    main()
