"""Accessible guided dashboard for LayerWise-QC.

This app is intentionally written as a guided research dashboard, not just a
collection of sliders and plots. Every major element explains:

- what it is,
- what input it uses,
- how to read it,
- what a good/bad value means,
- what limitation still exists.

The live decision still uses transparent demo/proxy scores. Real research claims
require real OT / MPM / PBI data and independent ground truth such as CT porosity,
relative density, metallography, or surface-defect labels.
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


# ---------------------------------------------------------------------------
# Page and display constants
# ---------------------------------------------------------------------------

STANDARD_VED = STANDARD_VED_J_MM3
LOW_LIMIT = STANDARD_VED * 0.82
HIGH_LIMIT = STANDARD_VED * 1.18

CLASS_DISPLAY = {
    "standard": {
        "short": "STABLE",
        "long": "stable process window",
        "meaning": "Energy input is close to the reference process window. In this prototype, that means low defect risk.",
        "operator": "Usually hold the parameters and continue monitoring.",
    },
    "delta_minus_30_ved": {
        "short": "LOW ENERGY",
        "long": "low-energy / lack-of-fusion risk",
        "meaning": "Energy input is too low. Powder may not fully melt, so lack-of-fusion type defects become more likely.",
        "operator": "Consider more energy input, or inspect powder spreading if PBI also looks abnormal.",
    },
    "delta_plus_30_ved": {
        "short": "HIGH ENERGY",
        "long": "high-energy / keyhole-spatter risk",
        "meaning": "Energy input is too high. The melt pool may become unstable, with keyhole, spatter, or overheating risk.",
        "operator": "Consider reducing energy input or allowing more cooling, but only after validation.",
    },
}

SENSOR_TEXT = {
    "ot": {
        "name": "Optical tomography",
        "simple": "Layer-wide thermal image.",
        "what": "OT captures broad layer-wise thermal emission. It is useful for heat accumulation and abnormal energy patterns.",
        "reads": "Bright or widespread intensity can suggest higher thermal emission or accumulated heat.",
        "limitation": "In this app, OT is simulated. Real OT needs calibration and alignment to the printed layer.",
    },
    "mpm": {
        "name": "Melt-pool monitoring",
        "simple": "Local melt-pool signal.",
        "what": "MPM focuses more on melt-pool behavior, local instability, bright spots, and local fusion disturbance.",
        "reads": "Sharp local intensity changes can suggest unstable melting, overheating, or local lack of fusion.",
        "limitation": "In this app, MPM is simulated. Real MPM depends on machine optics and sampling rate.",
    },
    "pbi": {
        "name": "Powder-bed imaging",
        "simple": "Powder/surface image.",
        "what": "PBI shows the powder bed or exposed surface. It is useful for recoater marks, streaks, powder shortage, and surface anomalies.",
        "reads": "Streaks or strong row-wise patterns can suggest poor spreading or recoater issues.",
        "limitation": "In this app, PBI is simulated. Real PBI requires lighting and viewpoint consistency.",
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
    show_guides: bool
    simple_mode: bool


# ---------------------------------------------------------------------------
# Generic UI guide helpers
# ---------------------------------------------------------------------------

def _apply_accessible_style() -> None:
    """Improve readability and hierarchy without requiring external CSS."""
    st.markdown(
        """
<style>
[data-testid="stMetricValue"] {
    font-size: 2.1rem;
}
[data-testid="stMetricLabel"] {
    font-size: 1.02rem;
}
.guided-card {
    border: 1px solid rgba(128, 128, 128, 0.35);
    border-radius: 0.75rem;
    padding: 1rem 1.1rem;
    margin: 0.6rem 0 1.0rem 0;
    background: rgba(128, 128, 128, 0.06);
}
.guide-title {
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 0.35rem;
}
.small-muted {
    opacity: 0.75;
    font-size: 0.92rem;
}
.good-box {
    border-left: 5px solid #3fb950;
    padding: 0.8rem 1rem;
    background: rgba(63, 185, 80, 0.10);
    margin: 0.4rem 0;
}
.warn-box {
    border-left: 5px solid #d29922;
    padding: 0.8rem 1rem;
    background: rgba(210, 153, 34, 0.10);
    margin: 0.4rem 0;
}
.bad-box {
    border-left: 5px solid #f85149;
    padding: 0.8rem 1rem;
    background: rgba(248, 81, 73, 0.10);
    margin: 0.4rem 0;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _guide_card(
    title: str,
    purpose: str,
    inputs: str,
    output: str,
    how_to_read: str,
    limitation: str,
    *,
    expanded: bool = True,
) -> None:
    """Reusable explanation block for each tab or component."""
    if not st.session_state.get("show_guides", True):
        return

    if expanded:
        st.markdown(
            f"""
<div class="guided-card">
<div class="guide-title">Guide — {title}</div>

**Purpose:** {purpose}

**Inputs used:** {inputs}

**What it gives:** {output}

**How to read it:** {how_to_read}

**Limitation:** {limitation}
</div>
            """.strip(),
            unsafe_allow_html=True,
        )
    else:
        with st.expander(f"Guide — {title}", expanded=False):
            st.markdown(
                f"""
**Purpose:** {purpose}

**Inputs used:** {inputs}

**What it gives:** {output}

**How to read it:** {how_to_read}

**Limitation:** {limitation}
                """.strip()
            )


def _chart_guide(title: str, how_to_read: str, good_bad: str) -> None:
    if not st.session_state.get("show_guides", True):
        return
    st.caption(f"How to read this chart — {title}: {how_to_read} {good_bad}")


def _table_guide(title: str, how_to_read: str) -> None:
    if not st.session_state.get("show_guides", True):
        return
    st.caption(f"How to read this table — {title}: {how_to_read}")


def _plain_info(title: str, text: str, kind: str = "info") -> None:
    if kind == "success":
        st.success(f"**{title}:** {text}")
    elif kind == "warning":
        st.warning(f"**{title}:** {text}")
    elif kind == "error":
        st.error(f"**{title}:** {text}")
    else:
        st.info(f"**{title}:** {text}")


def _formula_box(title: str, formula: str, meaning: str) -> None:
    if not st.session_state.get("show_guides", True):
        return
    st.markdown(
        f"""
<div class="guided-card">
<div class="guide-title">{title}</div>

```text
{formula}
```

{meaning}
</div>
        """.strip(),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

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


def _risk_badge(score: float) -> tuple[str, str, str]:
    if score < 0.35:
        return "LOW", "Process is currently close to the stable window.", "success"
    if score < 0.60:
        return "WATCH", "The layer is not failing in the demo, but it is moving away from the stable window.", "warning"
    return "HIGH", "The current settings strongly push the process toward a defect-prone region.", "error"


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
            spot_size_um=inputs.spot_size_um,
        )
    elif modality == "mpm":
        tuned = ProcessInputs(
            inputs.laser_power_w,
            inputs.scan_speed_mm_s,
            inputs.hatch_distance_mm,
            inputs.layer_thickness_mm,
            heat_memory=min(1.0, inputs.heat_memory + 0.12),
            powder_uniformity=inputs.powder_uniformity,
            spot_size_um=inputs.spot_size_um,
        )
    elif modality == "pbi":
        tuned = ProcessInputs(
            inputs.laser_power_w,
            inputs.scan_speed_mm_s,
            inputs.hatch_distance_mm,
            inputs.layer_thickness_mm,
            heat_memory=max(0.0, inputs.heat_memory - 0.04),
            powder_uniformity=max(0.0, inputs.powder_uniformity - 0.18),
            spot_size_um=inputs.spot_size_um,
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
        ("Laser power P", current.laser_power_w, reference.laser_power_w, "Higher P raises VED. Too high may push toward overheating/keyhole risk."),
        ("Scan speed v", current.scan_speed_mm_s, reference.scan_speed_mm_s, "Higher v lowers VED because the laser spends less time per distance."),
        ("Hatch distance h", current.hatch_distance_mm, reference.hatch_distance_mm, "Higher h lowers track overlap and lowers energy per volume."),
        ("Layer thickness t", current.layer_thickness_mm, reference.layer_thickness_mm, "Higher t spreads the same energy through more material and lowers VED."),
        ("Heat memory", current.heat_memory, reference.heat_memory, "Higher heat memory means the layer may behave hotter than VED alone suggests."),
        ("Powder uniformity", current.powder_uniformity, reference.powder_uniformity, "Lower uniformity suggests powder spreading or recoater risk."),
    ]

    for name, value, base, explanation in items:
        change = 0.0 if base == 0 else (value - base) / base * 100.0
        movement = "almost unchanged" if abs(change) < 1 else f"{change:+.1f}%"
        rows.append(
            {
                "parameter": name,
                "current": round(float(value), 4),
                "change vs preset": movement,
                "plain meaning": explanation,
            }
        )

    return pd.DataFrame(rows)


def _decision_summary_frame(
    state: ControlState,
    ot_scores: dict[str, float],
    second_scores: dict[str, float],
    fused: dict[str, float],
) -> pd.DataFrame:
    pred = prediction(fused)
    uncertainty, uncertainty_reason = compute_uncertainty(fused, ot_scores, second_scores)
    risk_score = risk_index(fused)

    return pd.DataFrame(
        [
            {
                "result": "Final condition",
                "value": _label(pred),
                "what it means": CLASS_DISPLAY[pred]["meaning"],
            },
            {
                "result": "Operator interpretation",
                "value": CLASS_DISPLAY[pred]["operator"],
                "what it means": "This is an advisory interpretation, not a machine command.",
            },
            {
                "result": "Risk index",
                "value": f"{risk_score:.2f}",
                "what it means": "Combined low-energy and high-energy score. Higher means more unstable in this demo.",
            },
            {
                "result": "Uncertainty",
                "value": uncertainty,
                "what it means": uncertainty_reason,
            },
            {
                "result": "Current VED ratio",
                "value": f"{state.inputs.ved / STANDARD_VED:.2f}× reference",
                "what it means": "Below 0.82× tends toward low energy; above 1.18× tends toward high energy in this demo.",
            },
        ]
    )


def _make_feature_export_row(state: ControlState) -> pd.DataFrame:
    """Create one-row export preview from current sidebar state."""
    physics = compute_physics_features(state.inputs)
    sensors = _current_sensor_descriptors(state)

    row: dict[str, float | str] = {
        "laser_power_w": state.inputs.laser_power_w,
        "scan_speed_mm_s": state.inputs.scan_speed_mm_s,
        "hatch_distance_mm": state.inputs.hatch_distance_mm,
        "layer_thickness_mm": state.inputs.layer_thickness_mm,
        "heat_memory": state.inputs.heat_memory,
        "powder_uniformity": state.inputs.powder_uniformity,
        "second_modality": state.second_modality,
    }

    row.update({f"phys_{k}": v for k, v in physics.items()})

    for modality, desc in sensors.items():
        row.update({f"{modality}_{k}": v for k, v in desc.items()})

    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

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
    ax.text(STANDARD_VED, 0.07, "reference\n37.78", ha="center", va="bottom", fontsize=9)
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
            laser_power_w=inputs.laser_power_w,
            scan_speed_mm_s=float(s),
            hatch_distance_mm=inputs.hatch_distance_mm,
            layer_thickness_mm=inputs.layer_thickness_mm,
            heat_memory=inputs.heat_memory,
            powder_uniformity=inputs.powder_uniformity,
            spot_size_um=inputs.spot_size_um,
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
    ax.set_title("Sensitivity check: what happens if only scan speed changes?", loc="left")
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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar_controls() -> ControlState:
    st.sidebar.title("Controls")
    st.sidebar.caption("Use this sidebar to create a process condition and see how the dashboard responds.")

    st.sidebar.subheader("Display")
    show_guides = st.sidebar.toggle(
        "Show explanations and guide text",
        value=True,
        help="Turn this on for teaching/review mode. Turn it off for compact dashboard mode.",
    )
    simple_mode = st.sidebar.toggle(
        "Simple wording mode",
        value=True,
        help="Keeps wording beginner-friendly and avoids too much jargon.",
    )
    st.session_state["show_guides"] = show_guides

    preset_name = st.sidebar.selectbox(
        "Choose a starting scenario",
        list(PRESETS.keys()),
        index=0,
        help="Presets are examples. You can still change every value manually.",
    )
    p0 = PRESETS[preset_name]

    with st.sidebar.expander("1) Process parameters", expanded=True):
        st.markdown(
            """
These are the machine/process inputs used to calculate VED.

- More **power** usually means more energy.
- More **speed**, **hatch distance**, or **layer thickness** usually lowers energy density.
- **Spot size** does not change VED, but it changes beam area and power density.
            """.strip()
        )
        laser_power = st.slider("Laser power P [W]", 180, 430, int(p0.laser_power_w), 1)
        scan_speed = st.slider("Scan speed v [mm/s]", 650, 1700, int(p0.scan_speed_mm_s), 1)
        hatch = st.slider("Hatch distance h [mm]", 0.07, 0.18, float(p0.hatch_distance_mm), 0.005)
        layer = st.slider("Layer thickness t [mm]", 0.02, 0.09, float(p0.layer_thickness_mm), 0.005)
        spot_size = st.slider("Laser spot size / beam diameter [µm]", 30, 200, int(p0.spot_size_um), 1)
        if spot_size < 40 or spot_size > 150:
            st.caption("Note: this spot size is outside the common demo range. That may be valid for a specific machine, but document the source.")

    with st.sidebar.expander("2) Process state proxies", expanded=True):
        st.markdown(
            """
These are **demo state variables**, not direct machine settings.

- **Heat memory** means the part may already be hot from previous layers.
- **Powder uniformity** means how good the powder spreading is.
            """.strip()
        )
        heat_memory = st.slider("Heat memory", 0.0, 1.0, float(p0.heat_memory), 0.01)
        powder_uniformity = st.slider("Powder-bed uniformity", 0.0, 1.0, float(p0.powder_uniformity), 0.01)

    with st.sidebar.expander("3) Sensor fusion setup", expanded=True):
        st.markdown(
            """
The app always uses **OT** as the layer-wide thermal signal.

Choose one extra sensor:
- **MPM**: melt-pool monitoring, useful for local melt-pool instability.
- **PBI**: powder-bed imaging, useful for recoater marks and powder spreading problems.
            """.strip()
        )

        second_modality = st.radio(
            "Choose second sensor",
            ["mpm", "pbi"],
            format_func=lambda x: {
                "mpm": "MPM — melt-pool monitoring",
                "pbi": "PBI — powder-bed imaging",
            }[x],
        )

        st.markdown("Choose how much the final decision should trust **OT** compared with the second sensor.")

        w_ot = st.slider(
            "Trust in OT",
            0.0,
            1.0,
            0.50,
            0.05,
            help="0.50 means OT and the second sensor have equal influence. Higher values trust OT more.",
        )
        w_second = 1.0 - w_ot

        st.info(f"Final score = {w_ot:.2f} × OT + {w_second:.2f} × {second_modality.upper()}")

    return ControlState(
        inputs=ProcessInputs(
            laser_power_w=laser_power,
            scan_speed_mm_s=scan_speed,
            hatch_distance_mm=hatch,
            layer_thickness_mm=layer,
            heat_memory=heat_memory,
            powder_uniformity=powder_uniformity,
            spot_size_um=spot_size,
        ),
        second_modality=second_modality,
        w_ot=w_ot,
        w_second=w_second,
        preset_name=preset_name,
        show_guides=show_guides,
        simple_mode=simple_mode,
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def _show_overview_tab(state: ControlState) -> None:
    st.header("Overview — what this dashboard does")

    _guide_card(
        "Overview",
        "Show the full workflow from process inputs to quality decision and next-layer advisory.",
        "Sidebar process settings, sensor choice, and fusion weight.",
        "A map of what the app calculates and what each tab is for.",
        "Read this tab first. It explains what is real, what is simulated, and what the final outputs mean.",
        "This is still a prototype. It does not prove real defect detection until real sensor data and ground truth are connected.",
    )

    st.subheader("One-line purpose")
    st.info(
        "LayerWise-QC estimates whether the current LPBF layer is stable, low-energy-risk, or high-energy-risk, "
        "then explains why and suggests whether the next layer should keep or adjust parameters."
    )

    st.subheader("Workflow")
    st.code(
        """1. User sets process parameters: P, v, h, t
2. App calculates VED and physics-informed descriptors
3. App generates/reads OT + MPM/PBI sensor signals
4. App extracts sensor descriptors and proxy/model scores
5. App fuses OT with MPM or PBI
6. App reports decision, uncertainty, explanation, and next-layer advisory""",
        language="text",
    )

    st.subheader("Current maturity")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data", "Synthetic/demo", help="The current app uses generated example images.")
    c2.metric("Model", "Proxy logic", help="The live score is transparent logic, not a loaded trained checkpoint.")
    c3.metric("Control", "Advisory", help="The app recommends actions; it does not command a machine.")
    c4.metric("Claim level", "Prototype", help="A serious claim needs real data and validation.")

    st.subheader("Outputs you get")
    _table_guide(
        "outputs",
        "Each row is one thing the dashboard produces. The second column explains why it matters.",
    )
    st.dataframe(overview_outputs_frame(), hide_index=True, use_container_width=True)

    st.subheader("What not to overclaim")
    st.warning(
        "Do not present this as a validated defect detector yet. Present it as a prototype architecture for "
        "physics-informed sensor fusion and feed-forward advisory control."
    )


def _show_process_inputs_tab(state: ControlState) -> None:
    st.header("Process inputs — what the sliders mean")

    _guide_card(
        "Process inputs",
        "Explain every input so the app is not just a black box.",
        "Laser power, scan speed, hatch distance, layer thickness, heat memory, and powder uniformity.",
        "A clear interpretation of how each input pushes the process toward stable, low-energy, or high-energy behavior.",
        "Look at the 'plain meaning' column. It tells you the physical direction of each input.",
        "Heat memory and powder uniformity are prototype sliders; real versions should come from layer history and PBI/thermal data.",
    )

    _formula_box(
        "Main energy equation used in the demo",
        "VED = laser power / (scan speed × hatch distance × layer thickness)",
        "Higher VED usually means more energy per printed volume. Low VED can cause lack of fusion; high VED can cause keyhole/spatter risk.",
    )

    st.subheader("Current input explanations")
    _table_guide(
        "input explanations",
        "Each row explains one sidebar control and what it usually does physically.",
    )
    st.dataframe(explain_process_inputs(state.inputs), hide_index=True, use_container_width=True)

    st.subheader("Change relative to selected scenario")
    _table_guide(
        "movement vs preset",
        "This table compares your current slider values to the preset you selected. Large changes explain why the risk output changes.",
    )
    st.dataframe(
        _parameter_effect_rows(state.inputs, PRESETS[state.preset_name]),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Quick interpretation of current VED")
    ved_ratio = state.inputs.ved / STANDARD_VED
    if ved_ratio < 0.82:
        _plain_info("Current VED interpretation", f"VED is {ved_ratio:.2f}× reference, below the stable window. This pushes toward low-energy/lack-of-fusion risk.", "warning")
    elif ved_ratio > 1.18:
        _plain_info("Current VED interpretation", f"VED is {ved_ratio:.2f}× reference, above the stable window. This pushes toward high-energy/keyhole-spatter risk.", "warning")
    else:
        _plain_info("Current VED interpretation", f"VED is {ved_ratio:.2f}× reference, inside the demo stable window.", "success")


def _show_live_decision_tab(state: ControlState) -> None:
    st.header("Live decision — what the app currently predicts")

    _guide_card(
        "Live decision",
        "Show the current quality state and explain why the app produced it.",
        "Current process settings, OT score, second-sensor score, and fusion weight.",
        "VED, rule state, fused condition, risk index, uncertainty, charts, and written reasons.",
        "Start with the big metrics, then read the explanation table. The chart shows whether the decision came from VED, sensors, or fusion.",
        "Scores are probability-like demo scores, not calibrated probabilities from a validated model.",
    )

    inputs = state.inputs
    ot_scores, second_scores, fused = _current_scores(state)
    rule_state = classify_from_ved(inputs.ved)
    fused_state = prediction(fused)
    risk_score = risk_index(fused)
    badge, badge_msg, badge_kind = _risk_badge(risk_score)
    uncertainty, uncertainty_reason = compute_uncertainty(fused, ot_scores, second_scores)

    st.subheader("Main result")
    physics_now = compute_physics_features(inputs)
    m1, m2, m3, m4, m5, m6, m7 = st.columns([1.0, 1.0, 1.0, 1.1, 1.1, 1.0, 1.0])
    m1.metric("VED", f"{inputs.ved:.2f}", "J/mm³", help="Energy density calculated from P/(v×h×t).")
    m2.metric("VED ratio", f"{inputs.ved / STANDARD_VED:.2f}×", help="Current VED divided by the reference VED.")
    m3.metric("Power density", f"{physics_now['power_density_w_mm2']:.0f}", "W/mm²", help="Laser power divided by beam area. VED alone does not capture this concentration.")
    m4.metric("VED rule", _label(rule_state, "short"), help="Simple rule-based state using only VED.")
    m5.metric("Fused output", _label(fused_state, "short"), help="Final state after combining OT and the second sensor.")
    m6.metric("Risk index", f"{risk_score:.2f}", badge, help="Low-energy score + high-energy score.")
    m7.metric("Uncertainty", uncertainty, help="High if sensors disagree or top scores are close.")

    _plain_info("Decision in words", f"{_label(fused_state)}. {CLASS_DISPLAY[fused_state]['meaning']}", badge_kind)
    st.caption(f"Uncertainty reason: {uncertainty_reason}")

    st.subheader("Decision summary table")
    _table_guide(
        "decision summary",
        "This table converts the numerical output into plain language. Read from top to bottom.",
    )
    st.dataframe(
        _decision_summary_frame(state, ot_scores, second_scores, fused),
        hide_index=True,
        use_container_width=True,
    )

    left, right = st.columns([0.52, 0.48], gap="large")

    with left:
        st.subheader("VED gauge")
        _chart_guide(
            "VED gauge",
            "The vertical line marked 'current' is your current energy density.",
            "Left side means low-energy risk; middle is the demo stable window; right side means high-energy risk.",
        )
        st.pyplot(_plot_ved_gauge(inputs.ved), use_container_width=True)

    with right:
        st.subheader("Score comparison")
        _chart_guide(
            "score comparison",
            "Each row is a possible condition; each bar shows how strongly OT, the second sensor, and fusion support it.",
            "The longest fused bar is the final decision. Similar bar lengths mean uncertainty.",
        )
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
    _table_guide(
        "explanation steps",
        "Each row is one reason used to interpret the current state. This is the main explainability section.",
    )
    st.dataframe(explanation_frame(reasons), hide_index=True, use_container_width=True)

    st.subheader("Sensitivity check")
    _chart_guide(
        "scan-speed sensitivity",
        "This plot changes scan speed only and keeps everything else fixed.",
        "If speed increases, energy density usually drops; if speed decreases, energy density usually rises.",
    )
    st.pyplot(_plot_sensitivity(state.inputs, state.second_modality, state.w_ot, state.w_second), use_container_width=True)


def _show_sensor_signals_tab(state: ControlState) -> None:
    st.header("Sensor signals — what OT, MPM, and PBI mean")

    _guide_card(
        "Sensor signals",
        "Explain the sensor images and their extracted descriptors.",
        "Simulated OT plus selected MPM or PBI image generated from the current process state.",
        "Sensor images, sensor roles, and descriptor table.",
        "Do not just look at the image. Read the descriptor table to see what the app extracts numerically.",
        "The current images are generated demo patches. Real images may require calibration, alignment, and preprocessing.",
    )

    ot_scores, second_scores, fused = _current_scores(state)
    images = _current_demo_images(state)
    descriptors = _current_sensor_descriptors(state)

    st.subheader("Sensor meaning")
    sensor_rows = []
    for key in ["ot", "mpm", "pbi"]:
        sensor_rows.append(
            {
                "sensor": key.upper(),
                "simple meaning": SENSOR_TEXT[key]["simple"],
                "what it detects": SENSOR_TEXT[key]["what"],
                "how to read it": SENSOR_TEXT[key]["reads"],
                "limitation": SENSOR_TEXT[key]["limitation"],
            }
        )
    _table_guide(
        "sensor meaning",
        "This table explains what each sensor is supposed to contribute before looking at the images.",
    )
    st.dataframe(pd.DataFrame(sensor_rows), hide_index=True, use_container_width=True)

    st.subheader("Current demo sensor images")
    cols = st.columns([0.30, 0.30, 0.40], gap="large")
    with cols[0]:
        st.markdown("**OT patch**")
        st.image(images["ot"], use_container_width=True)
        st.caption("OT is the always-on thermal layer signal in this demo.")
        st.write(f"OT proxy prediction: **{class_label(prediction(ot_scores))}**")

    with cols[1]:
        st.markdown(f"**{state.second_modality.upper()} patch**")
        st.image(images[state.second_modality], use_container_width=True)
        st.caption(f"This is the selected second sensor: {SENSOR_TEXT[state.second_modality]['name']}.")
        st.write(f"{state.second_modality.upper()} proxy prediction: **{class_label(prediction(second_scores))}**")

    with cols[2]:
        st.markdown("**What to notice**")
        if state.second_modality == "mpm":
            st.info("With MPM selected, the second sensor mainly represents local melt-pool behavior and hot/cold spots.")
        else:
            st.info("With PBI selected, the second sensor mainly represents powder spreading, streaks, and recoater-type issues.")
        st.write("The image is not the final decision. The decision comes after extracting descriptors and fusing sensor scores.")

    st.subheader("Sensor-derived descriptor table")
    _table_guide(
        "sensor descriptors",
        "Each descriptor is a numeric summary of the image. Mean/mode describe brightness; IQR/variance describe spread; texture/streakiness describe non-uniformity.",
    )
    st.dataframe(sensor_descriptors_to_frame(descriptors), hide_index=True, use_container_width=True)

    st.subheader("Plain-language descriptor guide")
    guide = pd.DataFrame(
        [
            {"descriptor family": "mean / median / mode", "meaning": "Overall brightness or dominant intensity."},
            {"descriptor family": "IQR / variance / standard deviation", "meaning": "How non-uniform or spread out the image intensities are."},
            {"descriptor family": "hot/cold pixel fraction", "meaning": "How much of the image is unusually bright or unusually dark."},
            {"descriptor family": "texture energy", "meaning": "How much local variation appears in the image."},
            {"descriptor family": "streakiness", "meaning": "Row-wise variation, especially useful for powder-bed/recoater patterns."},
        ]
    )
    st.dataframe(guide, hide_index=True, use_container_width=True)


def _show_physics_features_tab(state: ControlState) -> None:
    st.header("Physics-informed features — why this is not only black-box ML")

    _guide_card(
        "Physics-informed features",
        "Show physically meaningful descriptors used to make the prototype more research-facing.",
        "Current process inputs and demo state variables.",
        "A table of VED, normalized VED, linear energy, areal energy, beam/spot-size power density, geometry ratios, residence-time proxy, and heat-accumulation proxy.",
        "These features are the bridge between simple process parameters and data-driven modelling. They can be exported and used in ML baselines.",
        "These are simplified descriptors, not a full thermal simulation or finite-element model.",
    )

    features = compute_physics_features(state.inputs)

    st.subheader("Main feature table")
    _table_guide(
        "physics features",
        "Read the 'why it matters' column. It explains why each variable can help a model generalize beyond raw sliders.",
    )
    st.dataframe(physics_features_to_frame(features), hide_index=True, use_container_width=True)

    st.subheader("Most important current features")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Normalized VED", f"{features['normalized_ved']:.2f}", help="1.00 means equal to the reference process.")
    c2.metric("Linear energy", f"{features['linear_energy_j_mm']:.3f} J/mm", help="Laser power divided by scan speed.")
    c3.metric("Power density", f"{features['power_density_w_mm2']:.0f} W/mm²", help="Laser power divided by beam area from spot size.")
    c4.metric("Hatch/spot", f"{features['hatch_to_spot_ratio']:.2f}", help="Hatch distance divided by spot diameter; high values imply weaker track overlap.")
    st.info("VED alone does not capture beam concentration. Spot size changes beam area, power density, track overlap proxies, and therefore melt-pool behaviour even at the same VED.")

    st.subheader("Why this matters for research")
    st.info(
        "A serious model should compare process-only, sensor-only, image-only, and hybrid physics-informed versions. "
        "This table gives the process-feature side of that comparison."
    )

    st.subheader("Feature export preview for current slider state")
    _table_guide(
        "feature export preview",
        "This one-row table shows what could be exported to a CSV for model training or ablation.",
    )
    export_preview = _make_feature_export_row(state)
    st.dataframe(export_preview, use_container_width=True)

    csv = export_preview.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download current feature row as CSV",
        data=csv,
        file_name="current_layer_features.csv",
        mime="text/csv",
    )


def _show_sensor_fusion_tab(state: ControlState) -> None:
    st.header("Sensor fusion — how OT and the second sensor are combined")

    _guide_card(
        "Sensor fusion",
        "Explain how the app combines multiple sensor opinions.",
        "OT score, MPM/PBI score, and fusion weight selected in the sidebar.",
        "Fusion formula, agreement table, uncertainty level, and ablation model plan.",
        "If OT and the second sensor agree, the decision is stronger. If they disagree, uncertainty rises.",
        "This is a transparent weighted average. A future version can use a trained fusion model.",
    )

    ot_scores, second_scores, fused = _current_scores(state)
    uncertainty, uncertainty_reason = compute_uncertainty(fused, ot_scores, second_scores)

    st.subheader("Fusion formula")
    st.code(
        f"Final score = {state.w_ot:.2f} × OT score + {state.w_second:.2f} × {state.second_modality.upper()} score",
        language="text",
    )
    st.caption(
        "Example: if OT weight is 0.70, the final decision trusts OT more than the second sensor."
    )

    st.subheader("Agreement table")
    _table_guide(
        "sensor agreement",
        "Each row is one source. Compare the predicted condition and score columns. Disagreement means uncertainty should be treated seriously.",
    )
    st.dataframe(
        sensor_agreement_table(ot_scores, second_scores, fused, state.second_modality),
        hide_index=True,
        use_container_width=True,
    )

    if uncertainty == "HIGH":
        _plain_info("Uncertainty", uncertainty_reason, "error")
    elif uncertainty == "MEDIUM":
        _plain_info("Uncertainty", uncertainty_reason, "warning")
    else:
        _plain_info("Uncertainty", uncertainty_reason, "success")

    st.subheader("Common fusion cases")
    cases = pd.DataFrame(
        [
            {"case": "OT stable + PBI bad", "interpretation": "Possible powder spreading problem that thermal layer signal alone may miss."},
            {"case": "OT hot + MPM hot", "interpretation": "Likely high-energy or thermal accumulation behavior."},
            {"case": "OT low + MPM low", "interpretation": "Likely low-energy/lack-of-fusion condition."},
            {"case": "Sensors disagree", "interpretation": "Raise uncertainty and inspect raw sensor data before acting."},
        ]
    )
    st.dataframe(cases, hide_index=True, use_container_width=True)

    st.subheader("Ablation-ready model modes")
    _table_guide(
        "ablation modes",
        "This table explains how the project can compare simple and advanced models later.",
    )
    st.dataframe(ablation_modes_frame(), hide_index=True, use_container_width=True)


def _show_feedforward_tab(state: ControlState) -> None:
    st.header("Feed-forward control advisory — what to do next")

    _guide_card(
        "Feed-forward control",
        "Translate the current monitoring result into a next-layer recommendation.",
        "Fused risk state, VED, process settings, and sensor descriptors.",
        "A recommended next-layer action: hold, increase energy, decrease energy, or inspect powder-bed condition.",
        "Compare current power/speed with recommended power/speed. Zero change means the app recommends holding current settings.",
        "This is advisory only. It does not send commands to a machine and must be validated experimentally.",
    )

    _, _, fused = _current_scores(state)
    descriptors = _current_sensor_descriptors(state)
    rec = recommend_feedforward_control(state.inputs, fused, sensor_descriptors=descriptors)

    ved_ratio = state.inputs.ved / STANDARD_VED

    st.subheader("Current process condition")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current VED", f"{state.inputs.ved:.2f} J/mm³", help="Current volumetric energy density.")
    c2.metric("VED ratio", f"{ved_ratio:.2f}× reference", help="Current VED divided by reference VED.")
    c3.metric("Risk mode", rec.risk_mode, help="Detected process condition for the advisory decision.")
    c4.metric("Risk strength", f"{rec.confidence:.2f}", help="Combined non-stable risk signal, not model certainty.")

    st.subheader("Recommendation in plain words")
    if rec.risk_mode == "stable":
        _plain_info(
            "Recommended action",
            "The current process is closest to the stable window. Keep laser power and scan speed unchanged for the next layer.",
            "success",
        )
    elif "low-energy" in rec.risk_mode:
        _plain_info(
            "Recommended action",
            "Low-energy / lack-of-fusion risk dominates. Increase energy input conservatively for the next layer.",
            "warning",
        )
    elif "high-energy" in rec.risk_mode:
        _plain_info(
            "Recommended action",
            "High-energy / keyhole-spatter risk dominates. Decrease energy input conservatively for the next layer.",
            "warning",
        )
    elif "powder" in rec.risk_mode:
        _plain_info(
            "Recommended action",
            "Powder-bed or recoating risk is suspected. Inspect powder spreading before changing laser parameters.",
            "error",
        )
    else:
        _plain_info("Recommended action", rec.action, "info")

    st.subheader("Current vs recommended next-layer settings")
    st.info(
        "This table compares the current machine settings with the recommended next-layer settings. "
        "If the process is stable, the recommended values stay the same. If risk dominates, the app suggests a conservative change."
    )

    a, b = st.columns(2)
    with a:
        st.metric(
            "Laser power recommendation",
            f"{rec.recommended_power_w:.1f} W",
            f"{rec.delta_power_percent:+.2f}%",
            help="Recommended laser power for the next layer.",
        )
        st.caption(f"Current laser power: {rec.current_power_w:.1f} W")

    with b:
        st.metric(
            "Scan-speed recommendation",
            f"{rec.recommended_scan_speed_mm_s:.1f} mm/s",
            f"{rec.delta_scan_speed_percent:+.2f}%",
            help="Recommended scan speed for the next layer.",
        )
        st.caption(f"Current scan speed: {rec.current_scan_speed_mm_s:.1f} mm/s")

    _table_guide(
        "recommendation table",
        "Rows show current settings, recommended next-layer settings, correction size, rationale, and caution.",
    )
    detail = recommendation_to_frame(rec).copy()
    detail["item"] = detail["item"].replace({"confidence": "risk strength"})
    st.dataframe(detail, hide_index=True, use_container_width=True)

    st.subheader("Why this action was selected")
    st.info(rec.rationale)

    st.subheader("Important caution")
    st.warning(rec.caution)

    st.subheader("Control logic used by the prototype")
    st.code(
        """if low-energy / lack-of-fusion risk dominates:
    increase laser power slightly
    or reduce scan speed slightly

if high-energy / keyhole-spatter risk dominates:
    reduce laser power slightly
    or increase scan speed slightly

if powder-bed descriptors look abnormal:
    inspect recoating / powder spreading
    do not compensate only by laser power

if stable:
    hold current parameters""",
        language="text",
    )


def _show_data_manifest_tab() -> None:
    st.header("Data / manifest — how real data should enter the app")

    _guide_card(
        "Data / manifest",
        "Show what data structure is needed to move from demo to real research.",
        "Built-in demo manifest or uploaded CSV manifest.",
        "Dataset summary, missing-column check, class balance, and research-readiness checklist.",
        "A good manifest links every layer/region to process parameters, sensor image paths, and ground-truth quality labels.",
        "This tab validates data structure only. It does not train a model.",
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

    st.subheader("Expected columns")
    left, right = st.columns(2)
    with left:
        st.markdown("**Minimum columns for the current pipeline**")
        st.code(",".join(MINIMUM_COLUMNS), language="text")
    with right:
        st.markdown("**Recommended research columns**")
        st.code(",".join(RESEARCH_COLUMNS), language="text")

    if manifest.empty:
        st.warning("No manifest is currently loaded.")
        return

    ok, missing = validate_manifest_columns(manifest)
    if ok:
        st.success("Minimum manifest columns are present.")
    else:
        st.error(f"Missing required columns: {missing}")

    st.subheader("Manifest summary")
    _table_guide(
        "manifest summary",
        "This table tells you whether the dataset has enough metadata for layer-wise research.",
    )
    st.dataframe(manifest_summary(manifest), hide_index=True, use_container_width=True)

    st.subheader("Class balance")
    _table_guide(
        "class balance",
        "Imbalanced classes can make model metrics misleading. A serious study should report this.",
    )
    st.dataframe(class_balance(manifest), hide_index=True, use_container_width=True)

    st.subheader("Research readiness")
    _table_guide(
        "research readiness",
        "Rows marked 'no' are missing pieces needed for serious validation.",
    )
    st.dataframe(research_readiness_frame(manifest), hide_index=True, use_container_width=True)

    st.subheader("Missing image report")
    st.caption("This check is meaningful for repo files. Uploaded CSV paths may not exist inside Streamlit Cloud.")
    st.dataframe(
        missing_image_report(manifest, ROOT / "data" / "demo_samples"),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Manifest preview")
    st.dataframe(manifest.head(20), use_container_width=True)


def _show_sample_library_tab() -> None:
    st.header("Sample library — built-in demo examples")

    _guide_card(
        "Sample library",
        "Browse the built-in synthetic/demo examples.",
        "Demo manifest rows and generated OT/MPM/PBI image files.",
        "Side-by-side sensor patches for a selected condition.",
        "Use this to understand how the data path is supposed to look before real lab images are connected.",
        "These images are not experimental measurements.",
    )

    manifest = _read_manifest()
    if manifest is None or manifest.empty:
        st.info("No demo sample manifest found. Run `python scripts/make_synthetic_dataset.py --out data/demo_samples --layers 18`.")
        return

    class_pick = st.selectbox("Choose sample condition", CLASS_NAMES, format_func=lambda x: _label(x), key="sample_condition")
    subset = manifest[manifest["class_name"] == class_pick].reset_index(drop=True)
    row_number = st.slider("Sample row", 0, max(0, len(subset) - 1), 0) if len(subset) > 1 else 0
    row = subset.iloc[row_number] if len(subset) else manifest.iloc[0]

    st.write(f"**Selected condition:** {_label(class_pick)}.")
    st.caption(CLASS_DISPLAY[class_pick]["meaning"])

    cols = st.columns(3)
    for col, modality in zip(cols, ["ot", "mpm", "pbi"]):
        path_col = f"{modality}_path"
        with col:
            st.subheader(modality.upper())
            if path_col in row and isinstance(row[path_col], str):
                path = _resolve_demo_path(row[path_col])
                if path.exists():
                    st.image(Image.open(path), use_container_width=True)
                    st.caption(SENSOR_TEXT[modality]["what"])
                else:
                    st.warning(f"Missing {path.name}")
            else:
                st.warning("not in manifest")

    st.subheader("Selected manifest row")
    st.dataframe(pd.DataFrame([row]), use_container_width=True)


def _show_layer_history_tab() -> None:
    st.header("Layer history — why one bad layer can matter")

    _guide_card(
        "Layer history",
        "Explain why layer-wise monitoring matters over time.",
        "Selected disturbed layer range, disturbance type, and healing-capacity assumption.",
        "A conceptual timeline showing whether the defect-like signal decays or remains.",
        "If the disturbed stack is deeper than the healing capacity, residual risk remains in the sketch.",
        "This is a conceptual sketch, not a calibrated thermal or metallurgical model.",
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

    _chart_guide(
        "layer-history sketch",
        "The shaded area is the disturbed layer range. The line after that shows whether the risk decays or remains.",
        "If the disturbance lasts longer than the healing capacity, the residual signal stays higher.",
    )
    fig, msg = _plot_layer_story(start, n_layers, mode, healing_capacity)
    st.pyplot(fig, use_container_width=True)

    if n_layers > healing_capacity:
        st.warning(msg)
    else:
        st.success(msg)


def _show_validation_roadmap_tab() -> None:
    st.header("Validation roadmap — how to make this scientifically serious")

    _guide_card(
        "Validation roadmap",
        "List the steps needed before claiming real defect detection or real control.",
        "Current prototype status and future real data/ground truth requirements.",
        "A step-by-step research plan.",
        "Rows are ordered from easiest near-term coding work to real experimental validation.",
        "The roadmap is a plan, not proof of performance.",
    )

    roadmap = pd.DataFrame(
        [
            {
                "step": 1,
                "task": "Clarify app explanations",
                "coding output": "Guide cards, chart captions, table explanations, and plain-language summaries.",
                "research value": "Makes the prototype understandable and defensible.",
            },
            {
                "step": 2,
                "task": "Connect real data",
                "coding output": "Manifest with real OT, MPM, PBI paths and process metadata.",
                "research value": "Moves beyond generated examples.",
            },
            {
                "step": 3,
                "task": "Add real ground truth",
                "coding output": "Columns for CT porosity, relative density, metallography, or surface labels.",
                "research value": "Allows actual quality prediction.",
            },
            {
                "step": 4,
                "task": "Export feature table",
                "coding output": "CSV with process features, sensor descriptors, image-model scores, and labels.",
                "research value": "Enables reproducible ML experiments.",
            },
            {
                "step": 5,
                "task": "Train and compare baselines",
                "coding output": "Process-only, sensor-only, image-only, and hybrid models.",
                "research value": "Shows whether sensor fusion really helps.",
            },
            {
                "step": 6,
                "task": "Use grouped validation",
                "coding output": "Build-wise or specimen-wise split option.",
                "research value": "Avoids random-patch leakage.",
            },
            {
                "step": 7,
                "task": "Validate feed-forward policy",
                "coding output": "Repeated builds or simulation comparing corrected vs uncorrected layers.",
                "research value": "Tests whether recommendations improve quality.",
            },
        ]
    )
    st.dataframe(roadmap, hide_index=True, use_container_width=True)

    st.success(
        "Best current claim: a guided prototype architecture for physics-informed sensor fusion "
        "and feed-forward advisory control in PBF-LB."
    )




def _show_sample_model_requirements_tab() -> None:
    st.header("Sample / model needs — what to collect and how to train")

    _guide_card(
        "Sample and model requirements",
        "List exactly what must be collected for each LPBF sample and explain the five model types needed for training.",
        "Printed sample metadata, process parameters, quality labels, sensor files, and extracted features.",
        "A practical checklist for lab data collection, physics-informed feature building, model training, and validation.",
        "Start from the required sample table. Then check which model is possible depending on the available data.",
        "This tab is a planning and training checklist. It does not train the models directly.",
    )

    st.subheader("1. Minimum number of samples")

    sample_count = pd.DataFrame(
        [
            ["Minimum starting dataset", "50–80 samples", "Enough for a first physics-informed baseline and simple pass/fail model."],
            ["Good project dataset", "100–150 samples", "Better for train/test split, feature importance, and comparing model types."],
            ["Strong research dataset", "200–300 samples", "Better for DOE coverage, model validation, and publishable-level comparison."],
            ["Best transferability dataset", "Same samples on two machines or two process windows", "Useful for testing whether the physics-informed model generalizes."],
        ],
        columns=["Dataset level", "Recommended amount", "Meaning"],
    )
    st.dataframe(sample_count, hide_index=True, use_container_width=True)

    st.info(
        "For the first real model, the minimum practical target is 50–80 printed samples with process parameters and measured density or porosity."
    )

    st.subheader("2. Required information for each printed sample")

    sample_inputs = pd.DataFrame(
        [
            ["sample_id", "Unique sample name, for example S001", "Required"],
            ["build_id", "Build/job identifier, for example Build_01", "Required"],
            ["machine_id", "LPBF machine name or ID", "Required"],
            ["printing_date", "Date of printing", "Recommended"],
            ["material", "Material name, for example 316L, Ti64, AlSi10Mg", "Required"],
            ["powder_batch", "Powder batch or supplier batch number", "Recommended"],
            ["powder_reuse_number", "How many times the powder was reused", "If available"],
            ["sample_geometry", "Cube, cylinder, tensile bar, etc.", "Required"],
            ["sample_position_x", "X-position on the build plate", "Required"],
            ["sample_position_y", "Y-position on the build plate", "Required"],
            ["laser_power_W", "Laser power in watts", "Required"],
            ["scan_speed_mm_s", "Scan speed in mm/s", "Required"],
            ["hatch_spacing_um", "Hatch spacing in micrometers", "Required"],
            ["layer_thickness_um", "Layer thickness in micrometers", "Required"],
            ["spot_size_um", "Laser spot size or beam diameter", "Required"],
            ["scan_strategy", "Stripe, chessboard, island, rotation angle, contour strategy, etc.", "Required"],
            ["build_plate_temperature_C", "Build plate or preheating temperature", "If available"],
            ["oxygen_level_ppm", "Oxygen level during printing", "If available"],
            ["gas_flow_condition", "Gas flow direction or setting", "If available"],
            ["measured_density_percent", "Final relative density of the printed sample", "Required"],
            ["porosity_percent", "Porosity percentage from CT/microscopy/image analysis", "Recommended"],
            ["measurement_method", "Archimedes, micro-CT, microscopy, image analysis, etc.", "Required"],
            ["defect_type", "good, lack_of_fusion, keyhole, crack, delamination, failed, unknown", "Recommended"],
            ["failure_note", "Crack, collapse, delamination, rough surface, recoater issue, etc.", "Required"],
            ["sensor_file_name", "Pyrometer, OT, MPM, PBI, thermal camera, or layer-image file name", "If available"],
            ["general_notes", "Any extra observation during printing, removal, cleaning, or inspection", "Recommended"],
        ],
        columns=["Column name", "What to record", "Priority"],
    )
    st.dataframe(sample_inputs, hide_index=True, use_container_width=True)

    st.subheader("3. Sensor data needed if available")

    sensor_inputs = pd.DataFrame(
        [
            ["pyrometer_file", "Raw pyrometer or IR signal file", "Sensor-only and hybrid models"],
            ["thermal_camera_file", "Thermal camera recording or layer images", "Sensor-only and hybrid models"],
            ["melt_pool_monitoring_file", "MPM / coaxial melt-pool monitoring file", "Sensor-only and hybrid models"],
            ["optical_tomography_file", "OT layer-wise thermal/optical emission file", "Sensor-only and hybrid models"],
            ["powder_bed_image_file", "PBI image file for powder-bed/recoater condition", "Sensor-only and hybrid models"],
            ["layer_log_file", "Layer-wise machine/process log", "Layer-wise modelling"],
            ["machine_log_file", "Machine warning, error, oxygen, gas flow, or interruption log", "Data cleaning and failure explanation"],
            ["thermal_mean", "Average sensor signal", "Extracted feature"],
            ["thermal_max", "Maximum sensor signal", "Extracted feature"],
            ["thermal_std", "Thermal fluctuation", "Extracted feature"],
            ["thermal_iqr", "Interquartile range; melt-pool stability descriptor", "Extracted feature"],
            ["thermal_mode", "Most frequent thermal value", "Extracted feature"],
            ["thermal_skewness", "Asymmetry of thermal distribution", "Extracted feature"],
            ["thermal_kurtosis", "Outlier/sharpness behaviour", "Extracted feature"],
            ["hotspot_fraction", "Fraction of overheated area", "Extracted feature"],
            ["streakiness", "Row-wise pattern, useful for powder-bed/recoater problems", "Extracted feature"],
            ["number_of_anomalous_layers", "Number of suspicious layers in the sample", "Extracted feature"],
        ],
        columns=["Sensor item", "Meaning", "Used for"],
    )
    st.dataframe(sensor_inputs, hide_index=True, use_container_width=True)

    st.subheader("4. Physics-informed features calculated by the app")

    st.caption(
        "The lab does not need to manually calculate these. The app can calculate them from the raw process parameters and material properties."
    )

    physics_inputs = pd.DataFrame(
        [
            ["LED", "P / v", "Linear energy density"],
            ["AED", "P / (v × h)", "Areal energy density"],
            ["VED", "P / (v × h × t)", "Volumetric energy density"],
            ["normalized_VED", "η × VED / (ρ × Cp × ΔT)", "Energy normalized by material heating need"],
            ["hatch_to_spot_ratio", "h / spot_size", "Track overlap indicator"],
            ["hatch_to_layer_ratio", "h / t", "Hatch spacing relative to layer thickness"],
            ["spot_to_layer_ratio", "spot_size / t", "Beam size relative to layer thickness"],
            ["thermal_diffusion_ratio", "α / (v × spot_size)", "Heat diffusion compared with laser movement"],
            ["heat_accumulation_proxy", "function(VED, heat memory, layer history)", "Risk of accumulated heat"],
            ["thermal_stability_score", "function(IQR, std, mode)", "Sensor-based melt-pool stability"],
        ],
        columns=["Feature", "Formula / source", "Physical meaning"],
    )
    st.dataframe(physics_inputs, hide_index=True, use_container_width=True)

    st.subheader("5. Five models to train and what each one needs")

    models = pd.DataFrame(
        [
            [
                "Model 1 — Process-only baseline",
                "laser_power_W, scan_speed_mm_s, hatch_spacing_um, layer_thickness_um, spot_size_um, scan_strategy",
                "measured_density_percent or porosity_percent",
                "Linear Regression, Random Forest, Gradient Boosting, Gaussian Process",
                "Shows what can be predicted from raw machine settings only. This is the baseline.",
            ],
            [
                "Model 2 — Physics-informed model",
                "LED, AED, VED, normalized_VED, hatch_to_spot_ratio, hatch_to_layer_ratio, spot_to_layer_ratio, thermal_diffusion_ratio",
                "measured_density_percent or porosity_percent",
                "Random Forest, Gradient Boosting, Gaussian Process, Symbolic Regression",
                "More explainable model based on energy input, track overlap, and heat-transfer-related descriptors.",
            ],
            [
                "Model 3 — Sensor-only model",
                "thermal_mean, thermal_max, thermal_std, thermal_iqr, thermal_mode, skewness, kurtosis, hotspot_fraction, streakiness, anomalous_layers",
                "density, porosity, defect class, or pass/fail label",
                "Random Forest, XGBoost/Gradient Boosting, SVM, simple CNN later if enough images exist",
                "In-situ monitoring model. Useful when pyrometry, OT, MPM, PBI, or thermal images are available.",
            ],
            [
                "Model 4 — Hybrid physics + sensor model",
                "process parameters + physics-informed features + sensor descriptors",
                "density, porosity, defect class, or pass/fail label",
                "Gradient Boosting, XGBoost, Random Forest, Stacking model, Gaussian Process",
                "Strongest model because it combines process design, physics meaning, and real monitoring data.",
            ],
            [
                "Model 5 — Pass/fail or defect classification model",
                "process features, physics features, and sensor features if available",
                "pass/fail or defect_type such as good, lack_of_fusion, keyhole, crack, failed",
                "Logistic Regression, Random Forest Classifier, SVM, Gradient Boosting Classifier",
                "Best first classification model for small datasets. Easier than predicting many defect classes.",
            ],
        ],
        columns=["Model", "Inputs needed", "Target/output needed", "ML algorithms", "Purpose"],
    )
    st.dataframe(models, hide_index=True, use_container_width=True)

    st.subheader("6. ML instruments and tools needed")

    instruments = pd.DataFrame(
        [
            ["Clean dataset table", "One CSV/Excel row per sample", "Needed before any model training"],
            ["Unit checker", "Convert μm to mm where needed and check impossible values", "Prevents wrong VED/feature values"],
            ["Feature builder", "Automatically calculate LED, AED, VED, normalized VED, ratios, and stability features", "Creates physics-informed inputs"],
            ["Train/test split", "70/30 split or 80/20 split", "Basic validation"],
            ["Cross-validation", "5-fold cross-validation, especially for 50–80 samples", "More reliable small-dataset evaluation"],
            ["Grouped split", "Split by build_id if several samples come from the same build", "Prevents leakage between train and test"],
            ["Scaler", "StandardScaler or MinMaxScaler", "Needed for SVM, linear models, and Gaussian Process"],
            ["Missing-value handler", "Drop, fill, or flag missing sensor/material values", "Keeps model training stable"],
            ["Feature selection", "RFECV, permutation importance, correlation filtering", "Removes weak or duplicate features"],
            ["Metrics for regression", "R², MAE, RMSE", "Used for density/porosity prediction"],
            ["Metrics for classification", "Accuracy, precision, recall, F1, confusion matrix, ROC-AUC if binary", "Used for pass/fail or defect classification"],
            ["Explainability", "SHAP, permutation importance, feature importance", "Explains why the model predicts good or bad quality"],
            ["Ablation comparison", "Process-only vs physics-only vs sensor-only vs hybrid", "Proves whether physics/sensors improve the model"],
            ["Model export", "joblib or pickle file", "Allows the trained model to be loaded into the Streamlit app"],
            ["Prediction interface", "Input fields + feature calculation + trained model output", "Turns the trained model into a usable dashboard tool"],
        ],
        columns=["ML instrument/tool", "What it is", "Why it is needed"],
    )
    st.dataframe(instruments, hide_index=True, use_container_width=True)

    st.subheader("7. Recommended training order")

    st.markdown(
        """
        1. Collect 50–80 samples minimum with process parameters and measured density/porosity.  
        2. Clean the table and check all units.  
        3. Calculate physics-informed features: LED, AED, VED, normalized VED, and geometric ratios.  
        4. Train **Model 1: process-only baseline**.  
        5. Train **Model 2: physics-informed model**.  
        6. If sensor data exists, extract thermal descriptors and train **Model 3: sensor-only model**.  
        7. Combine all features and train **Model 4: hybrid model**.  
        8. Convert density/porosity into pass/fail labels and train **Model 5: classification model**.  
        9. Compare all models using R², MAE, RMSE, F1, confusion matrix, and feature importance.  
        10. Export the best model and connect it back to the LayerWise-QC dashboard.
        """
    )

    st.success(
        "Practical first goal: train the process-only and physics-informed models first, then improve the app with sensor-only and hybrid models when real OT/MPM/PBI or pyrometry data is available."
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="LayerWise-QC", page_icon="🧩", layout="wide")
    _apply_accessible_style()

    state = _sidebar_controls()

    st.title("LayerWise-QC dashboard")
    st.caption("Guided physics-informed sensor fusion and feed-forward advisory prototype for PBF-LB.")

    if state.show_guides:
        st.info(
            "Guide mode is ON. Every tab includes explanations for what the section does, how to read charts/tables, and what the limits are."
        )

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
            "sample/model needs",
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

    with tabs[11]:
        _show_sample_model_requirements_tab()


if __name__ == "__main__":
    main()
