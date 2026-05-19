"""Interactive dashboard for LayerWise-QC.

The app is a front-end for explaining and testing a layer-wise quality-monitoring
logic before real sensor images are connected. It combines a transparent VED rule,
modality-specific sensor proxies, fusion weights, simulated sensor patches, and a
layer-history sketch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import textwrap

import matplotlib.pyplot as plt
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

STANDARD_VED = 37.78
LOW_LIMIT = STANDARD_VED * 0.82
HIGH_LIMIT = STANDARD_VED * 1.18

CLASS_DISPLAY = {
    "standard": {
        "short": "STABLE",
        "long": "stable process window",
        "meaning": "The energy input is close to the reference window. The layer is treated as low risk in this demo.",
    },
    "delta_minus_30_ved": {
        "short": "LOW ENERGY",
        "long": "low-energy / lack-of-fusion risk",
        "meaning": "Energy input is too low. Powder may not fully melt, so lack-of-fusion type defects become more likely.",
    },
    "delta_plus_30_ved": {
        "short": "HIGH ENERGY",
        "long": "high-energy / keyhole-spatter risk",
        "meaning": "Energy input is too high. The melt pool may become unstable, with keyhole, spatter, or overheating risk.",
    },
}

SENSOR_TEXT = {
    "ot": {
        "name": "Optical tomography",
        "what": "Global layer-wise thermal emission. It is useful for broad heat accumulation and abnormal energy patterns.",
        "demo": "In this demo, OT becomes brighter or more distributed when heat memory is high.",
    },
    "mpm": {
        "name": "Melt-pool monitoring",
        "what": "More local melt-pool signal. It is useful for local instability, bright spots, and local fusion disturbances.",
        "demo": "In this demo, MPM reacts more sharply to local low-energy gaps or high-energy spots.",
    },
    "pbi": {
        "name": "Powder-bed imaging",
        "what": "Image of the spread powder or exposed layer surface. It is useful for recoater marks, powder shortage, streaks, and surface anomalies.",
        "demo": "In this demo, PBI becomes streakier when powder-bed uniformity is reduced.",
    },
}

PRESETS = {
    "reference / stable": ProcessInputs(340, 1250, 0.12, 0.06, heat_memory=0.35, powder_uniformity=0.86),
    "too cold / lack-of-fusion risk": ProcessInputs(238, 1250, 0.12, 0.06, heat_memory=0.26, powder_uniformity=0.76),
    "too hot / keyhole-spatter risk": ProcessInputs(370, 1046.38, 0.12, 0.06, heat_memory=0.72, powder_uniformity=0.82),
    "bad powder spread": ProcessInputs(330, 1280, 0.12, 0.06, heat_memory=0.38, powder_uniformity=0.48),
}


@dataclass(frozen=True)
class ControlState:
    inputs: ProcessInputs
    second_modality: str
    w_ot: float
    w_second: float
    preset_name: str


# ----------------------------- small utilities -----------------------------


def _label(name: str, mode: str = "long") -> str:
    return CLASS_DISPLAY[name][mode]


def _wrap(s: str, width: int = 32) -> str:
    return "\n".join(textwrap.wrap(s, width=width))


def _bar_df(scores: dict[str, float], source: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state": [_label(k) for k in CLASS_NAMES],
            "score": [scores[k] for k in CLASS_NAMES],
            "source": source,
        }
    )


def _prediction(scores: dict[str, float]) -> str:
    return max(scores, key=scores.get)


def _risk_index(scores: dict[str, float]) -> float:
    return float(scores["delta_minus_30_ved"] + scores["delta_plus_30_ved"])


def _risk_badge(score: float) -> tuple[str, str]:
    if score < 0.35:
        return "LOW", "Process is currently close to the stable window."
    if score < 0.60:
        return "WATCH", "The layer is not failing in the demo, but it is moving away from the stable window."
    return "HIGH", "The current settings strongly push the process toward a defect-prone region."


def _read_manifest() -> pd.DataFrame | None:
    path = ROOT / "data" / "demo_samples" / "manifest.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _resolve_demo_path(rel: str) -> Path:
    return ROOT / "data" / "demo_samples" / rel


def _modality_scores(inputs: ProcessInputs, modality: str) -> dict[str, float]:
    """Transparent sensor proxy for the dashboard.

    It is deliberately simple. The trained repository models still live in the
    training scripts; this function is only for the live dashboard where the user
    changes process settings without loading a checkpoint.
    """
    if modality == "ot":
        tuned = ProcessInputs(
            inputs.laser_power_w,
            inputs.scan_speed_mm_s,
            inputs.hatch_distance_mm,
            inputs.layer_thickness_mm,
            heat_memory=min(1.0, inputs.heat_memory + 0.08),
            powder_uniformity=min(1.0, inputs.powder_uniformity + 0.02),
        )
    elif modality == "mpm":
        tuned = ProcessInputs(
            inputs.laser_power_w,
            inputs.scan_speed_mm_s,
            inputs.hatch_distance_mm,
            inputs.layer_thickness_mm,
            heat_memory=min(1.0, inputs.heat_memory + 0.12),
            powder_uniformity=inputs.powder_uniformity,
        )
    elif modality == "pbi":
        tuned = ProcessInputs(
            inputs.laser_power_w,
            inputs.scan_speed_mm_s,
            inputs.hatch_distance_mm,
            inputs.layer_thickness_mm,
            heat_memory=max(0.0, inputs.heat_memory - 0.04),
            powder_uniformity=max(0.0, inputs.powder_uniformity - 0.18),
        )
    else:
        tuned = inputs
    return soft_process_scores(tuned)


def _parameter_effect_rows(current: ProcessInputs, reference: ProcessInputs) -> pd.DataFrame:
    rows = []
    items = [
        ("laser power P", current.laser_power_w, reference.laser_power_w, "Higher P raises VED and may move the process toward overheating."),
        ("scan speed v", current.scan_speed_mm_s, reference.scan_speed_mm_s, "Higher v lowers VED because the laser spends less time per distance."),
        ("hatch distance h", current.hatch_distance_mm, reference.hatch_distance_mm, "Higher h lowers overlap and lowers VED per volume."),
        ("layer thickness L", current.layer_thickness_mm, reference.layer_thickness_mm, "Higher L spreads the same energy through more material and lowers VED."),
        ("heat memory", current.heat_memory, reference.heat_memory, "Higher heat memory raises high-energy/thermal-accumulation risk."),
        ("powder uniformity", current.powder_uniformity, reference.powder_uniformity, "Lower uniformity raises powder-bed and lack-of-fusion risk."),
    ]
    for name, value, base, explanation in items:
        if base == 0:
            change = 0.0
        else:
            change = (value - base) / base * 100.0
        if abs(change) < 1:
            movement = "almost unchanged"
        elif change > 0:
            movement = f"+{change:.1f}%"
        else:
            movement = f"{change:.1f}%"
        rows.append({"parameter": name, "current": value, "change vs preset": movement, "meaning": explanation})
    return pd.DataFrame(rows)


def _explanation_rows(inputs: ProcessInputs, fused: dict[str, float], second_modality: str) -> pd.DataFrame:
    ratio = inputs.ved / STANDARD_VED
    rows = []
    if ratio < 0.82:
        rows.append(
            {
                "signal": "Energy density below window",
                "why it matters": f"VED is {ratio:.2f}× the reference. Low energy can leave un-melted zones.",
                "pushes output toward": _label("delta_minus_30_ved"),
            }
        )
    elif ratio > 1.18:
        rows.append(
            {
                "signal": "Energy density above window",
                "why it matters": f"VED is {ratio:.2f}× the reference. Excess energy can destabilize the melt pool.",
                "pushes output toward": _label("delta_plus_30_ved"),
            }
        )
    else:
        rows.append(
            {
                "signal": "Energy density inside window",
                "why it matters": f"VED is {ratio:.2f}× the reference, inside the demo window of 0.82–1.18×.",
                "pushes output toward": _label("standard"),
            }
        )
    if inputs.heat_memory > 0.62:
        rows.append(
            {
                "signal": "Heat memory is high",
                "why it matters": "Accumulated heat can make later layers brighter and less stable even when the nominal VED is not extreme.",
                "pushes output toward": _label("delta_plus_30_ved"),
            }
        )
    if inputs.powder_uniformity < 0.70:
        rows.append(
            {
                "signal": "Powder-bed uniformity is low",
                "why it matters": f"The {second_modality.upper()} channel is penalized because poor spreading can create local shortage or recoater marks.",
                "pushes output toward": _label("delta_minus_30_ved"),
            }
        )
    if not rows:
        rows.append(
            {
                "signal": "No strong disturbance",
                "why it matters": "The process parameters and sensor proxies are not far from the reference state.",
                "pushes output toward": _label(_prediction(fused)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------- charts ----------------------------------


def _plot_ved_gauge(ved: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 1.65))
    ax.set_xlim(18, 60)
    ax.set_ylim(0, 1)
    ax.axvspan(18, LOW_LIMIT, alpha=0.18, label="low-energy side")
    ax.axvspan(LOW_LIMIT, HIGH_LIMIT, alpha=0.28, label="stable window")
    ax.axvspan(HIGH_LIMIT, 60, alpha=0.18, label="high-energy side")
    ax.axvline(STANDARD_VED, linestyle="--", linewidth=2)
    ax.axvline(ved, linewidth=4)
    ax.text(ved, 0.72, f"current\n{ved:.1f}", ha="center", va="bottom", fontsize=10)
    ax.text(STANDARD_VED, 0.07, "reference 37.78", ha="center", va="bottom", fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("Volumetric energy density, J/mm³")
    ax.set_title("VED position: low energy → stable window → high energy", loc="left")
    ax.spines[["left", "right", "top"]].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_scores(score_table: pd.DataFrame) -> plt.Figure:
    pivot = score_table.pivot(index="state", columns="source", values="score")
    order = [_label(k) for k in CLASS_NAMES]
    pivot = pivot.reindex(order)
    wrapped = [_wrap(x, 26) for x in pivot.index]

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    y = np.arange(len(pivot.index))
    sources = list(pivot.columns)
    width = 0.22 if len(sources) == 3 else 0.30
    offsets = np.linspace(-width, width, len(sources))
    for offset, src in zip(offsets, sources):
        vals = pivot[src].values
        ax.barh(y + offset, vals, height=width * 0.88, label=src.upper())
        for yi, val in zip(y + offset, vals):
            ax.text(min(val + 0.015, 0.98), yi, f"{val:.2f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(wrapped)
    ax.set_xlim(0, 1)
    ax.set_xlabel("probability-like score")
    ax.set_title("Scores by source: OT vs second channel vs fused output", loc="left")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_sensitivity(inputs: ProcessInputs, second_modality: str, w_ot: float, w_second: float) -> plt.Figure:
    speeds = np.linspace(750, 1650, 26)
    rows = []
    for s in speeds:
        trial = ProcessInputs(
            inputs.laser_power_w,
            float(s),
            inputs.hatch_distance_mm,
            inputs.layer_thickness_mm,
            inputs.heat_memory,
            inputs.powder_uniformity,
        )
        fused = fuse_scores(_modality_scores(trial, "ot"), _modality_scores(trial, second_modality), w_ot, w_second)
        rows.append({"scan_speed": s, "stable": fused["standard"], "low_energy": fused["delta_minus_30_ved"], "high_energy": fused["delta_plus_30_ved"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.plot(df["scan_speed"], df["stable"], label="stable")
    ax.plot(df["scan_speed"], df["low_energy"], label="low-energy risk")
    ax.plot(df["scan_speed"], df["high_energy"], label="high-energy risk")
    ax.axvline(inputs.scan_speed_mm_s, linestyle="--", linewidth=2)
    ax.set_xlabel("scan speed v [mm/s]")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title("What happens if only scan speed changes?", loc="left")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_layer_story(start: int, n_layers: int, mode: str, healing_capacity: int) -> tuple[plt.Figure, pd.DataFrame, str]:
    xs = np.arange(0, 90)
    base = np.zeros_like(xs, dtype=float) + 0.06
    disturbed = (xs >= start) & (xs < start + n_layers)
    base[disturbed] = 0.75 if mode == "delta_minus_30_ved" else 0.62
    after = xs >= start + n_layers
    decay = np.exp(-0.25 * (xs[after] - (start + n_layers)))
    base[after] = 0.06 + (0.45 if n_layers > healing_capacity else 0.18) * decay
    if n_layers > healing_capacity:
        base[after] += min(0.22, (n_layers - healing_capacity) * 0.05)
    base = np.clip(base, 0, 1)

    df = pd.DataFrame({"layer": xs, "defect-like signal": base})
    fig, ax = plt.subplots(figsize=(8.8, 3.1))
    ax.plot(df["layer"], df["defect-like signal"], linewidth=2)
    ax.axvspan(start, start + n_layers, alpha=0.18, label="disturbed layers")
    ax.axvline(start + n_layers, linestyle="--", linewidth=1.5, label="standard exposure resumes")
    ax.set_ylim(0, 1)
    ax.set_xlabel("layer number")
    ax.set_ylabel("defect-like signal")
    ax.set_title("Layer history sketch", loc="left")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if n_layers > healing_capacity:
        msg = f"Disturbance depth is {n_layers} layers, above the selected healing capacity of {healing_capacity}. Residual risk is kept after normal exposure resumes."
    else:
        msg = f"Disturbance depth is {n_layers} layers, within the selected healing capacity of {healing_capacity}. The signal decays after normal exposure resumes."
    return fig, df, msg


# ------------------------------- UI sections -------------------------------


def _sidebar_controls() -> ControlState:
    st.sidebar.title("Controls")
    st.sidebar.caption("Change these values and watch VED, sensor scores, and the explanation update immediately.")

    preset_name = st.sidebar.selectbox("Process preset", list(PRESETS.keys()), index=0)
    p0 = PRESETS[preset_name]

    with st.sidebar.expander("Process parameters", expanded=True):
        laser_power = st.slider("Laser power P [W]", 180, 430, int(p0.laser_power_w), 1)
        scan_speed = st.slider("Scan speed v [mm/s]", 650, 1700, int(p0.scan_speed_mm_s), 1)
        hatch = st.slider("Hatch distance h [mm]", 0.07, 0.18, float(p0.hatch_distance_mm), 0.005)
        layer = st.slider("Layer thickness L [mm]", 0.02, 0.09, float(p0.layer_thickness_mm), 0.005)

    with st.sidebar.expander("Process memory / image condition", expanded=True):
        heat_memory = st.slider("Heat memory", 0.0, 1.0, float(p0.heat_memory), 0.01)
        powder_uniformity = st.slider("Powder-bed uniformity", 0.0, 1.0, float(p0.powder_uniformity), 0.01)

    with st.sidebar.expander("Fusion setup", expanded=True):
        second_modality = st.radio("Second channel", ["mpm", "pbi"], format_func=lambda x: x.upper(), horizontal=True)
        w_ot = st.slider("OT weight", 0.0, 1.0, 0.50, 0.05)
        w_second = 1.0 - w_ot
        st.caption(f"Second-channel weight: {w_second:.2f}")

    return ControlState(
        inputs=ProcessInputs(laser_power, scan_speed, hatch, layer, heat_memory, powder_uniformity),
        second_modality=second_modality,
        w_ot=w_ot,
        w_second=w_second,
        preset_name=preset_name,
    )


def _show_system_summary() -> None:
    st.title("LayerWise-QC dashboard")
    st.markdown(
        """
This page explains a layer-wise quality-monitoring workflow for laser powder-bed fusion. It is not just an image viewer: each change in the controls updates the energy-density calculation, the sensor-channel scores, the fused decision, and the written explanation.

**Reading the page:** low VED usually means lack-of-fusion risk; high VED usually means keyhole/spatter risk; poor powder spreading mainly affects the powder-bed imaging path.
        """.strip()
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.info("1. Set process parameters")
    c2.info("2. Inspect sensor proxies")
    c3.info("3. Fuse OT + second channel")
    c4.info("4. Read why the result changed")


def _show_decision_panel(state: ControlState) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    inputs = state.inputs
    ot_scores = _modality_scores(inputs, "ot")
    second_scores = _modality_scores(inputs, state.second_modality)
    fused = fuse_scores(ot_scores, second_scores, w_a=state.w_ot, w_b=state.w_second)
    rule_state = classify_from_ved(inputs.ved)
    fused_state = _prediction(fused)
    risk_score = _risk_index(fused)
    badge, badge_msg = _risk_badge(risk_score)

    st.header("1. Live decision")
    m1, m2, m3, m4, m5 = st.columns([1.1, 1.1, 1.2, 1.2, 1.0])
    m1.metric("VED", f"{inputs.ved:.2f}", "J/mm³")
    m2.metric("Reference ratio", f"{inputs.ved / STANDARD_VED:.2f}×", "vs 37.78")
    m3.metric("Rule state", _label(rule_state, "short"))
    m4.metric("Fused output", _label(fused_state, "short"))
    m5.metric("Risk index", f"{risk_score:.2f}", badge)

    st.write(f"**Current interpretation:** {_label(fused_state)}. {CLASS_DISPLAY[fused_state]['meaning']}")
    st.caption(badge_msg)

    left, right = st.columns([0.52, 0.48], gap="large")
    with left:
        st.pyplot(_plot_ved_gauge(inputs.ved), use_container_width=True)
    with right:
        score_table = pd.concat(
            [_bar_df(ot_scores, "ot"), _bar_df(second_scores, state.second_modality), _bar_df(fused, "fused")],
            ignore_index=True,
        )
        st.pyplot(_plot_scores(score_table), use_container_width=True)

    return ot_scores, second_scores, fused


def _show_why_panel(state: ControlState, fused: dict[str, float]) -> None:
    st.header("2. Why the output moved")
    st.markdown(
        """
The dashboard uses the standard VED relation:

`VED = laser power / (scan speed × hatch distance × layer thickness)`

So, increasing power raises energy density. Increasing scan speed, hatch distance, or layer thickness lowers it. Heat memory and powder uniformity are extra process-state sliders used to show how sensor channels can disagree even when VED is similar.
        """.strip()
    )

    ref = PRESETS[state.preset_name]
    a, b = st.columns([0.50, 0.50], gap="large")
    with a:
        st.subheader("Parameter movement")
        st.dataframe(_parameter_effect_rows(state.inputs, ref), hide_index=True, use_container_width=True)
    with b:
        st.subheader("Decision reasons")
        st.dataframe(_explanation_rows(state.inputs, fused, state.second_modality), hide_index=True, use_container_width=True)

    st.subheader("Sensitivity check")
    st.caption("This plot keeps all sliders fixed except scan speed. It shows why scan speed is a strong lever: faster scanning lowers VED, slower scanning raises VED.")
    st.pyplot(_plot_sensitivity(state.inputs, state.second_modality, state.w_ot, state.w_second), use_container_width=True)


def _show_sensor_panel(state: ControlState, ot_scores: dict[str, float], second_scores: dict[str, float], fused: dict[str, float]) -> None:
    st.header("3. Sensor view")
    st.write(
        "The images below are generated demo patches. They are not lab measurements. Their job is to make the data path understandable before real OT, MPM, or PBI files are connected."
    )

    seed = int(state.inputs.laser_power_w + state.inputs.scan_speed_mm_s + 1000 * state.inputs.hatch_distance_mm + 1000 * state.inputs.layer_thickness_mm)
    cols = st.columns([0.31, 0.31, 0.38], gap="large")
    with cols[0]:
        st.subheader("OT patch")
        st.image(image_from_process(state.inputs, PATCH_SIZES_HW["ot"], "ot", seed=seed), use_container_width=True)
        st.caption(SENSOR_TEXT["ot"]["what"])
        st.write(f"OT prediction: **{_label(_prediction(ot_scores))}**")
    with cols[1]:
        st.subheader(f"{state.second_modality.upper()} patch")
        st.image(
            image_from_process(state.inputs, PATCH_SIZES_HW[state.second_modality], state.second_modality, seed=seed + 3),
            use_container_width=True,
        )
        st.caption(SENSOR_TEXT[state.second_modality]["what"])
        st.write(f"{state.second_modality.upper()} prediction: **{_label(_prediction(second_scores))}**")
    with cols[2]:
        st.subheader("Fusion table")
        table = pd.concat(
            [_bar_df(ot_scores, "ot"), _bar_df(second_scores, state.second_modality), _bar_df(fused, "fused")],
            ignore_index=True,
        )
        pivot = table.pivot(index="state", columns="source", values="score").round(3)
        st.dataframe(pivot, use_container_width=True)
        st.markdown(
            f"""
**Fusion rule:** `{state.w_ot:.2f} × OT + {state.w_second:.2f} × {state.second_modality.upper()}`

The second channel is useful when it reacts to something OT does not capture strongly. For example, PBI reacts to powder uniformity, while MPM reacts more to local melt-pool disturbance.
            """.strip()
        )

    with st.expander("What each channel is supposed to add"):
        rows = []
        for key in ["ot", "mpm", "pbi"]:
            rows.append({"channel": key.upper(), "role": SENSOR_TEXT[key]["what"], "demo behavior": SENSOR_TEXT[key]["demo"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _show_sample_library() -> None:
    st.header("4. Built-in sample library")
    manifest = _read_manifest()
    if manifest is None or manifest.empty:
        st.info("No demo sample manifest found. Run `python scripts/make_synthetic_dataset.py --out data/demo_samples --layers 18`.")
        return

    class_pick = st.selectbox(
        "Choose sample condition",
        CLASS_NAMES,
        format_func=lambda x: _label(x),
        key="sample_condition",
    )
    subset = manifest[manifest["class_name"] == class_pick].reset_index(drop=True)
    row_number = st.slider("Sample row", 0, max(0, len(subset) - 1), 0) if len(subset) > 1 else 0
    row = subset.iloc[row_number] if len(subset) else manifest.iloc[0]

    st.write(f"**Selected condition:** {_label(class_pick)}. {CLASS_DISPLAY[class_pick]['meaning']}")
    cols = st.columns(3)
    for col, modality in zip(cols, ["ot", "mpm", "pbi"]):
        path_col = f"{modality}_path"
        with col:
            st.subheader(modality.upper())
            if path_col in row and isinstance(row[path_col], str):
                path = _resolve_demo_path(row[path_col])
                if path.exists():
                    st.image(Image.open(path), use_container_width=True)
                else:
                    st.warning(f"Missing {path.name}")
            else:
                st.warning("not in manifest")


def _show_layer_story() -> None:
    st.header("5. Layer history / healing sketch")
    st.write(
        "This part is a simple timeline sketch. It shows why layer-wise monitoring matters: a short disturbed region may disappear after standard exposure, while a deeper disturbed stack can leave residual risk."
    )
    a, b, c, d = st.columns(4)
    with a:
        n_layers = st.slider("Disturbed layers", 1, 12, 7)
    with b:
        start = st.slider("Disturbance starts at layer", 5, 70, 20)
    with c:
        healing_capacity = st.slider("Healing capacity assumption", 1, 10, 7)
    with d:
        mode = st.radio("Disturbance type", ["delta_minus_30_ved", "delta_plus_30_ved"], format_func=lambda x: _label(x, "short"))

    fig, df, msg = _plot_layer_story(start, n_layers, mode, healing_capacity)
    st.pyplot(fig, use_container_width=True)
    if n_layers > healing_capacity:
        st.warning(msg)
    else:
        st.success(msg)

    st.dataframe(
        pd.DataFrame(
            [
                {"item": "disturbed layer range", "value": f"{start} to {start + n_layers - 1}"},
                {"item": "standard exposure resumes", "value": f"layer {start + n_layers}"},
                {"item": "selected disturbance", "value": _label(mode)},
                {"item": "message", "value": msg},
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )


def _show_pipeline_panel() -> None:
    st.header("6. How this connects to the training code")
    st.markdown(
        """
The live dashboard uses transparent proxy scores so it can run without trained checkpoints. The repository also contains the actual training path:

1. prepare a CSV manifest with one row per layer/patch,
2. train one model for OT,
3. train one model for the second channel, either MPM or PBI,
4. fuse the output probabilities,
5. inspect metrics and Grad-CAM overlays.

Minimum real-data manifest columns:

```text
sample_id,class_idx,class_name,ot_path,mpm_path,pbi_path,laser_power_w,scan_speed_mm_s,hatch_distance_mm,layer_thickness_mm,ved_j_mm3
```

For a real experiment, the dashboard should be fed by actual layer-wise image patches and the trained checkpoints, not by the proxy score function used here.
        """.strip()
    )

    st.subheader("Current data flow")
    st.code(
        """process settings + layer image patches
        ↓
OT model                         second-channel model
        ↓                                  ↓
OT class scores                   MPM/PBI class scores
        \                                  /
         late fusion of probabilities
                  ↓
        layer-wise quality state + explanation""",
        language="text",
    )


def main() -> None:
    state = _sidebar_controls()
    _show_system_summary()

    tab_live, tab_sensors, tab_samples, tab_layers, tab_pipeline = st.tabs(
        ["live decision", "sensor view", "sample library", "layer history", "pipeline"]
    )

    with tab_live:
        ot_scores, second_scores, fused = _show_decision_panel(state)
        _show_why_panel(state, fused)
    with tab_sensors:
        ot_scores = _modality_scores(state.inputs, "ot")
        second_scores = _modality_scores(state.inputs, state.second_modality)
        fused = fuse_scores(ot_scores, second_scores, w_a=state.w_ot, w_b=state.w_second)
        _show_sensor_panel(state, ot_scores, second_scores, fused)
    with tab_samples:
        _show_sample_library()
    with tab_layers:
        _show_layer_story()
    with tab_pipeline:
        _show_pipeline_panel()


if __name__ == "__main__":
    main()
