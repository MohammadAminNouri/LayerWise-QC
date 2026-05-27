"""Interactive dashboard for LayerWise-QC.

This version makes the prototype easier to understand and closer to a serious
research direction. It separates the dashboard into clear steps:

1. overview,
2. process inputs,
3. live quality decision,
4. sensor signals,
5. physics-informed features,
6. sensor fusion and uncertainty,
7. feed-forward advisory control,
8. data / manifest readiness,
9. validation roadmap.

The live model still uses transparent proxy scores so the app can run without
trained checkpoints. Real deployment requires real OT / MPM / PBI images and
independent ground truth such as CT, density, metallography, or porosity labels.
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

from am_defect_detection.constants import CLASS_NAMES, PATCH_SIZES_HW  # noqa: E402
from am_defect_detection.data_manifest import (  # noqa: E402
    MINIMUM_COLUMNS,
    RESEARCH_COLUMNS,
    class_balance,
    manifest_summary,
    missing_image_report,
    research_readiness_frame,
    validate_manifest_columns,
)
from am_defect_detection.explainers import (  # noqa: E402
    explain_decision,
    explain_process_inputs,
    explanation_frame,
    overview_outputs_frame,
)
from am_defect_detection.feedforward_control import (  # noqa: E402
    recommend_feedforward_control,
    recommendation_to_frame,
)
from am_defect_detection.fusion_analysis import (  # noqa: E402
    ablation_modes_frame,
    compute_uncertainty,
    label as class_label,
    prediction,
    risk_index,
    sensor_agreement_table,
)
from am_defect_detection.physics_features import (  # noqa: E402
    STANDARD_VED_J_MM3,
    compute_physics_features,
    physics_features_to_frame,
)
from am_defect_detection.sensor_features import (  # noqa: E402
    compute_sensor_descriptors,
    sensor_descriptors_to_frame,
)
from am_defect_detection.simulation import (  # noqa: E402
    ProcessInputs,
    classify_from_ved,
    fuse_scores,
    image_from_process,
    soft_process_scores,
)


STANDARD_VED = STANDARD_VED_J_MM3
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
        "what": "Global layer-wise thermal emission. Useful for heat accumulation and broad abnormal energy patterns.",
        "demo": "In this demo, OT becomes brighter or more distributed when heat memory is high.",
    },
    "mpm": {
        "name": "Melt-pool monitoring",
        "what": "Local melt-pool signal. Useful for local instability, bright spots, and fusion disturbances.",
        "demo": "In this demo, MPM reacts sharply to local low-energy gaps or high-energy spots.",
    },
    "pbi": {
        "name": "Powder-bed imaging",
        "what": "Powder/surface image. Useful for recoater marks, powder shortage, streaks, and surface anomalies.",
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


def _label(name: str, mode: str = "long") -> str:
    return CLASS_DISPLAY[name][mode]


def _wrap(s: str, width: int = 32) -> str:
    return "\n".join(textwrap.wrap(s, width=width))


def _make_seed(inputs: ProcessInputs) -> int:
    return int(
        inputs.laser_power_w
        + inputs.scan_speed_mm_s
        + 1000 * inputs.hatch_distance_mm
        + 1000 * inputs.layer_thickness_mm
    )


def _bar_df(scores: dict[str, float], source: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state": [_label(k) for k in CLASS_NAMES],
            "score": [scores[k] for k in CLASS_NAMES],
            "source": source,
        }
    )


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
    """Transparent sensor proxy used only by the live demo dashboard."""
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


def _current_scores(state: ControlState) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    ot_scores = _modality_scores(state.inputs, "ot")
    second_scores = _modality_scores(state.inputs, state.second_modality)
    fused = fuse_scores(ot_scores, second_scores, w_a=state.w_ot, w_b=state.w_second)
    return ot_scores, second_scores, fused


def _current_demo_images(state: ControlState) -> dict[str, object]:
    seed = _make_seed(state.inputs)
    return {
        "ot": image_from_process(state.inputs, PATCH_SIZES_HW["ot"], "ot", seed=seed),
        state.second_modality: image_from_process(
            state.inputs,
            PATCH_SIZES_HW[state.second_modality],
            state.second_modality,
            seed=seed + 3,
        ),
    }


def _current_sensor_descriptors(state: ControlState) -> dict[str, dict[str, float]]:
    images = _current_demo_images(state)
    return {modality: compute_sensor_descriptors(img) for modality, img in images.items()}


def _parameter_effect_rows(current: ProcessInputs, reference: ProcessInputs) -> pd.DataFrame:
    rows = []
    items = [
        ("laser power P", current.laser_power_w, reference.laser_power_w, "Higher P raises VED and may move the process toward overheating."),
        ("scan speed v", current.scan_speed_mm_s, reference.scan_speed_mm_s, "Higher v lowers VED because the laser spends less time per distance."),
        ("hatch distance h", current.hatch_distance_mm, reference.hatch_distance_mm, "Higher h lowers overlap and lowers VED per volume."),
        ("layer thickness t", current.layer_thickness_mm, reference.layer_thickness_mm, "Higher t spreads the same energy through more material and lowers VED."),
        ("heat memory", current.heat_memory, reference.heat_memory, "Higher heat memory raises thermal-accumulation risk."),
        ("powder uniformity", current.powder_uniformity, reference.powder_uniformity, "Lower uniformity raises powder-bed and lack-of-fusion risk."),
    ]

    for name, value, base, explanation in items:
        change = 0.0 if base == 0 else (value - base) / base * 100.0
        movement = "almost unchanged" if abs(change) < 1 else f"{change:+.1f}%"
        rows.append(
            {
                "parameter": name,
                "current": value,
                "change vs preset": movement,
                "meaning": explanation,
            }
        )

    return pd.DataFrame(rows)


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
        fused = fuse_scores(
            _modality_scores(trial, "ot"),
            _modality_scores(trial, second_modality),
            w_ot,
            w_second,
        )
        rows.append(
            {
                "scan_speed": s,
                "stable": fused["standard"],
                "low_energy": fused["delta_minus_30_ved"],
                "high_energy": fused["delta_plus_30_ved"],
            }
        )

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


def _plot_layer_story(start: int, n_layers: int, mode: str, healing_capacity: int) -> tuple[plt.Figure, str]:
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

    return fig, msg


def _info_box(title: str, input_text: str, output_text: str, limitation: str) -> None:
    with st.expander(f"How to read this tab: {title}", expanded=False):
        st.markdown(
            f"""
**Input:** {input_text}

**Output:** {output_text}

**Limitation:** {limitation}
            """.strip()
        )


def _sidebar_controls() -> ControlState:
    st.sidebar.title("Controls")
    st.sidebar.caption("Change values and watch VED, sensor scores, physics features, and control advisory update.")

    preset_name = st.sidebar.selectbox("Process preset", list(PRESETS.keys()), index=0)
    p0 = PRESETS[preset_name]

    with st.sidebar.expander("Process parameters", expanded=True):
        laser_power = st.slider("Laser power P [W]", 180, 430, int(p0.laser_power_w), 1)
        scan_speed = st.slider("Scan speed v [mm/s]", 650, 1700, int(p0.scan_speed_mm_s), 1)
        hatch = st.slider("Hatch distance h [mm]", 0.07, 0.18, float(p0.hatch_distance_mm), 0.005)
        layer = st.slider("Layer thickness t [mm]", 0.02, 0.09, float(p0.layer_thickness_mm), 0.005)

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


def _show_overview_tab(state: ControlState) -> None:
    st.header("Overview: what the prototype does")
    st.markdown(
        """
This dashboard is a guided research prototype for layer-wise quality monitoring in laser powder-bed fusion.

It currently uses synthetic/demo sensor images and transparent proxy scores. The purpose is to explain the workflow before real OT, MPM, PBI, and ground-truth quality data are connected.
        """.strip()
    )

    st.subheader("Workflow")
    st.code(
        """process parameters: P, v, h, t
        ↓
VED + physics-informed descriptors
        ↓
OT / MPM / PBI sensor signals
        ↓
sensor descriptors + proxy/model scores
        ↓
sensor fusion + uncertainty
        ↓
layer-wise quality decision
        ↓
feed-forward advisory for the next layer""",
        language="text",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Current data", "Synthetic/demo")
    c2.metric("Current model", "Transparent proxy")
    c3.metric("Control", "Advisory only")

    st.subheader("What the app gives")
    st.dataframe(overview_outputs_frame(), hide_index=True, use_container_width=True)

    st.warning(
        "Safe scientific claim: this is a prototype architecture for physics-informed sensor fusion "
        "and feed-forward advisory control. It is not yet a validated defect-detection system."
    )


def _show_process_inputs_tab(state: ControlState) -> None:
    st.header("Process inputs")
    _info_box(
        "process inputs",
        "Laser power, scan speed, hatch distance, layer thickness, heat memory, and powder-bed uniformity.",
        "A clear explanation of what each input means and how it affects the risk logic.",
        "Heat memory and powder uniformity are demo sliders until real layer-history and powder-bed measurements are connected.",
    )

    st.dataframe(explain_process_inputs(state.inputs), hide_index=True, use_container_width=True)

    st.subheader("Movement versus selected preset")
    st.dataframe(
        _parameter_effect_rows(state.inputs, PRESETS[state.preset_name]),
        hide_index=True,
        use_container_width=True,
    )


def _show_live_decision_tab(state: ControlState) -> None:
    st.header("Live quality decision")
    _info_box(
        "live decision",
        "Current process settings and sensor-fusion weights.",
        "VED, rule-based state, fused class, risk index, uncertainty, and explanation.",
        "The live scores are transparent proxies, not trained model predictions.",
    )

    inputs = state.inputs
    ot_scores, second_scores, fused = _current_scores(state)
    rule_state = classify_from_ved(inputs.ved)
    fused_state = prediction(fused)
    risk_score = risk_index(fused)
    badge, badge_msg = _risk_badge(risk_score)
    uncertainty, uncertainty_reason = compute_uncertainty(fused, ot_scores, second_scores)

    m1, m2, m3, m4, m5, m6 = st.columns([1.0, 1.0, 1.1, 1.1, 1.0, 1.0])
    m1.metric("VED", f"{inputs.ved:.2f}", "J/mm³")
    m2.metric("Reference ratio", f"{inputs.ved / STANDARD_VED:.2f}×")
    m3.metric("Rule state", _label(rule_state, "short"))
    m4.metric("Fused output", _label(fused_state, "short"))
    m5.metric("Risk index", f"{risk_score:.2f}", badge)
    m6.metric("Uncertainty", uncertainty)

    st.write(f"**Interpretation:** {_label(fused_state)}. {CLASS_DISPLAY[fused_state]['meaning']}")
    st.caption(badge_msg)
    st.caption(f"Uncertainty reason: {uncertainty_reason}")

    left, right = st.columns([0.52, 0.48], gap="large")
    with left:
        st.pyplot(_plot_ved_gauge(inputs.ved), use_container_width=True)
    with right:
        score_table = pd.concat(
            [_bar_df(ot_scores, "ot"), _bar_df(second_scores, state.second_modality), _bar_df(fused, "fused")],
            ignore_index=True,
        )
        st.pyplot(_plot_scores(score_table), use_container_width=True)

    st.subheader("Why did the app say this?")
    descriptors = _current_sensor_descriptors(state)
    reasons = explain_decision(
        state.inputs,
        fused,
        descriptors,
        second_modality=state.second_modality,
    )
    st.dataframe(explanation_frame(reasons), hide_index=True, use_container_width=True)

    st.subheader("Scan-speed sensitivity")
    st.caption("All sliders are fixed except scan speed. Faster scanning lowers VED; slower scanning raises VED.")
    st.pyplot(_plot_sensitivity(state.inputs, state.second_modality, state.w_ot, state.w_second), use_container_width=True)


def _show_sensor_signals_tab(state: ControlState) -> None:
    st.header("Sensor signals")
    _info_box(
        "sensor signals",
        "Synthetic OT plus either MPM or PBI patch generated from the current process state.",
        "Images, descriptor table, and explanation of what each sensor contributes.",
        "The current images are demo patches, not lab measurements.",
    )

    ot_scores, second_scores, fused = _current_scores(state)
    images = _current_demo_images(state)
    descriptors = _current_sensor_descriptors(state)

    cols = st.columns([0.30, 0.30, 0.40], gap="large")
    with cols[0]:
        st.subheader("OT patch")
        st.image(images["ot"], use_container_width=True)
        st.caption(SENSOR_TEXT["ot"]["what"])
        st.write(f"OT prediction: **{class_label(prediction(ot_scores))}**")

    with cols[1]:
        st.subheader(f"{state.second_modality.upper()} patch")
        st.image(images[state.second_modality], use_container_width=True)
        st.caption(SENSOR_TEXT[state.second_modality]["what"])
        st.write(f"{state.second_modality.upper()} prediction: **{class_label(prediction(second_scores))}**")

    with cols[2]:
        st.subheader("Sensor roles")
        rows = [
            {"channel": key.upper(), "role": SENSOR_TEXT[key]["what"], "demo behavior": SENSOR_TEXT[key]["demo"]}
            for key in ["ot", "mpm", "pbi"]
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("Sensor-derived descriptors")
    st.dataframe(sensor_descriptors_to_frame(descriptors), hide_index=True, use_container_width=True)


def _show_physics_features_tab(state: ControlState) -> None:
    st.header("Physics-informed features")
    _info_box(
        "physics-informed features",
        "Current process inputs.",
        "Interpretable descriptors such as normalized VED, linear energy, areal energy, residence-time proxy, and heat-accumulation proxy.",
        "These are descriptors and proxies, not a complete finite-element or thermal-fluid model.",
    )

    features = compute_physics_features(state.inputs)
    st.dataframe(physics_features_to_frame(features), hide_index=True, use_container_width=True)

    st.subheader("Why this matters")
    st.info(
        "This tab answers: what makes the prototype physics-informed? "
        "The model is no longer only a black-box image classifier or VED gauge. "
        "It exposes physically meaningful descriptors that can be used in ablation studies."
    )


def _show_sensor_fusion_tab(state: ControlState) -> None:
    st.header("Sensor fusion and uncertainty")
    _info_box(
        "sensor fusion",
        "OT scores, MPM/PBI scores, and the selected fusion weight.",
        "A fused score, sensor agreement table, uncertainty level, and ablation logic.",
        "Current fusion is a transparent weighted average, not yet a trained fusion network.",
    )

    ot_scores, second_scores, fused = _current_scores(state)
    uncertainty, uncertainty_reason = compute_uncertainty(fused, ot_scores, second_scores)

    st.subheader("Fusion rule")
    st.code(
        f"fused score = {state.w_ot:.2f} × OT score + {state.w_second:.2f} × {state.second_modality.upper()} score",
        language="text",
    )

    st.subheader("Sensor agreement")
    st.dataframe(
        sensor_agreement_table(ot_scores, second_scores, fused, state.second_modality),
        hide_index=True,
        use_container_width=True,
    )

    if uncertainty == "HIGH":
        st.error(f"Uncertainty: {uncertainty}. {uncertainty_reason}")
    elif uncertainty == "MEDIUM":
        st.warning(f"Uncertainty: {uncertainty}. {uncertainty_reason}")
    else:
        st.success(f"Uncertainty: {uncertainty}. {uncertainty_reason}")

    st.subheader("Ablation-ready model modes")
    st.dataframe(ablation_modes_frame(), hide_index=True, use_container_width=True)


def _show_feedforward_tab(state: ControlState) -> None:
    st.header("Feed-forward control advisory")
    _info_box(
        "feed-forward control",
        "Fused risk state, process inputs, and sensor descriptors.",
        "A conservative next-layer recommendation: hold, increase energy, decrease energy, or inspect powder-bed condition.",
        "This is advisory only. It does not send commands to a machine.",
    )

    _, _, fused = _current_scores(state)
    descriptors = _current_sensor_descriptors(state)

    rec = recommend_feedforward_control(state.inputs, fused, sensor_descriptors=descriptors)

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk mode", rec.risk_mode)
    c2.metric("Power change", f"{rec.delta_power_percent:+.2f}%")
    c3.metric("Speed change", f"{rec.delta_scan_speed_percent:+.2f}%")

    st.dataframe(recommendation_to_frame(rec), hide_index=True, use_container_width=True)

    st.warning(
        "This is not automatic machine control. Real deployment needs controller access, "
        "safety limits, experimental validation, and quality ground truth."
    )

    st.subheader("Control logic")
    st.code(
        """if low-energy / lack-of-fusion risk dominates:
    increase laser power slightly
    or reduce scan speed slightly

if high-energy / keyhole-spatter risk dominates:
    reduce laser power slightly
    or increase scan speed slightly

if powder-bed descriptors look abnormal:
    inspect recoating / powder spreading
    do not compensate only by laser power""",
        language="text",
    )


def _show_data_manifest_tab() -> None:
    st.header("Data / manifest readiness")
    _info_box(
        "data readiness",
        "Built-in demo manifest or uploaded CSV manifest.",
        "Dataset summary, missing-column report, class balance, and research-readiness checklist.",
        "This tab does not train a model yet. It checks whether the data is structured correctly.",
    )

    uploaded = st.file_uploader("Upload a CSV manifest", type=["csv"])

    if uploaded is not None:
        manifest = pd.read_csv(uploaded)
        st.success("Uploaded manifest loaded.")
    else:
        manifest = _read_manifest()
        if manifest is None:
            st.info("No built-in demo manifest found. Upload a CSV or run the synthetic dataset script.")
            manifest = pd.DataFrame()

    if manifest.empty:
        st.subheader("Expected minimum columns")
        st.code(",".join(MINIMUM_COLUMNS), language="text")
        st.subheader("Recommended research columns")
        st.code(",".join(RESEARCH_COLUMNS), language="text")
        return

    ok, missing = validate_manifest_columns(manifest)
    if ok:
        st.success("Minimum manifest columns are present.")
    else:
        st.error(f"Missing required columns: {missing}")

    st.subheader("Manifest summary")
    st.dataframe(manifest_summary(manifest), hide_index=True, use_container_width=True)

    st.subheader("Class balance")
    st.dataframe(class_balance(manifest), hide_index=True, use_container_width=True)

    st.subheader("Research readiness")
    st.dataframe(research_readiness_frame(manifest), hide_index=True, use_container_width=True)

    st.subheader("Missing image report for built-in demo data")
    st.caption("This check is meaningful for repo files. Uploaded CSV paths may not exist inside Streamlit.")
    st.dataframe(
        missing_image_report(manifest, ROOT / "data" / "demo_samples"),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Preview")
    st.dataframe(manifest.head(20), use_container_width=True)


def _show_sample_library_tab() -> None:
    st.header("Built-in sample library")
    manifest = _read_manifest()
    if manifest is None or manifest.empty:
        st.info("No demo sample manifest found. Run `python scripts/make_synthetic_dataset.py --out data/demo_samples --layers 18`.")
        return

    class_pick = st.selectbox("Choose sample condition", CLASS_NAMES, format_func=lambda x: _label(x), key="sample_condition")
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


def _show_layer_history_tab() -> None:
    st.header("Layer history / healing sketch")
    _info_box(
        "layer history",
        "A user-selected disturbed layer range and healing-capacity assumption.",
        "A simple timeline showing whether residual risk decays or remains.",
        "This is a conceptual sketch, not a calibrated thermal-history model.",
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

    fig, msg = _plot_layer_story(start, n_layers, mode, healing_capacity)
    st.pyplot(fig, use_container_width=True)

    if n_layers > healing_capacity:
        st.warning(msg)
    else:
        st.success(msg)


def _show_validation_roadmap_tab() -> None:
    st.header("Validation roadmap")
    st.markdown(
        """
This tab explains what must be added before the prototype can become a serious research claim.
        """.strip()
    )

    roadmap = pd.DataFrame(
        [
            {
                "step": 1,
                "task": "Connect real data",
                "output": "Manifest with real OT, MPM, PBI paths and process metadata.",
            },
            {
                "step": 2,
                "task": "Add real ground truth",
                "output": "CT porosity, relative density, metallography, or surface-defect labels.",
            },
            {
                "step": 3,
                "task": "Export feature table",
                "output": "One CSV with process features, sensor descriptors, model scores, and labels.",
            },
            {
                "step": 4,
                "task": "Train baselines",
                "output": "Process-only, sensor-descriptor-only, image-only, and hybrid models.",
            },
            {
                "step": 5,
                "task": "Use grouped splits",
                "output": "Build-wise or specimen-wise validation to avoid leakage.",
            },
            {
                "step": 6,
                "task": "Validate feed-forward policy",
                "output": "Show that recommended corrections improve real quality metrics.",
            },
        ]
    )
    st.dataframe(roadmap, hide_index=True, use_container_width=True)

    st.success(
        "Best current claim: prototype architecture for physics-informed sensor fusion "
        "and feed-forward advisory control in PBF-LB."
    )


def main() -> None:
    st.set_page_config(page_title="LayerWise-QC", page_icon="🧩", layout="wide")

    state = _sidebar_controls()

    st.title("LayerWise-QC dashboard")
    st.caption("Physics-informed sensor fusion and feed-forward advisory prototype for PBF-LB.")

    tabs = st.tabs(
        [
            "overview",
            "process inputs",
            "live decision",
            "sensor signals",
            "physics features",
            "sensor fusion",
            "feed-forward control",
            "data / manifest",
            "sample library",
            "layer history",
            "validation roadmap",
        ]
    )

    with tabs[0]:
        _show_overview_tab(state)

    with tabs[1]:
        _show_process_inputs_tab(state)

    with tabs[2]:
        _show_live_decision_tab(state)

    with tabs[3]:
        _show_sensor_signals_tab(state)

    with tabs[4]:
        _show_physics_features_tab(state)

    with tabs[5]:
        _show_sensor_fusion_tab(state)

    with tabs[6]:
        _show_feedforward_tab(state)

    with tabs[7]:
        _show_data_manifest_tab()

    with tabs[8]:
        _show_sample_library_tab()

    with tabs[9]:
        _show_layer_history_tab()

    with tabs[10]:
        _show_validation_roadmap_tab()


if __name__ == "__main__":
    main()
