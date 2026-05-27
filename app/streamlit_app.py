"""Interactive dashboard for LayerWise-QC.

This app is a research-facing prototype for explaining and testing layer-wise
quality monitoring in laser powder-bed fusion before real sensor images and
trained checkpoints are connected.

Compared with the original dashboard, this version adds:

1. physics-informed process descriptors,
2. sensor-derived image descriptors,
3. explicit ablation logic for process-only, sensor-only, and fused reasoning,
4. layer-to-layer feed-forward control recommendations,
5. clearer research framing for future real-data validation.

The live decision path still uses transparent proxy scores so the dashboard can
run without trained checkpoints. The training scripts and real-data manifest can
later replace these proxy scores with trained OT / MPM / PBI models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
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
from am_defect_detection.simulation import (  # noqa: E402
    ProcessInputs,
    classify_from_ved,
    fuse_scores,
    image_from_process,
    soft_process_scores,
)


# ---------------------------------------------------------------------------
# Constants and display dictionaries
# ---------------------------------------------------------------------------

STANDARD_VED = 37.78
LOW_LIMIT = STANDARD_VED * 0.82
HIGH_LIMIT = STANDARD_VED * 1.18

REFERENCE_LASER_POWER_W = 340.0
REFERENCE_SCAN_SPEED_MM_S = 1250.0
REFERENCE_HATCH_DISTANCE_MM = 0.12
REFERENCE_LAYER_THICKNESS_MM = 0.06

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
        "what": "Local melt-pool signal. It is useful for local instability, bright spots, and local fusion disturbances.",
        "demo": "In this demo, MPM reacts more sharply to local low-energy gaps or high-energy spots.",
    },
    "pbi": {
        "name": "Powder-bed imaging",
        "what": "Image of the spread powder or exposed layer surface. It is useful for recoater marks, powder shortage, streaks, and surface anomalies.",
        "demo": "In this demo, PBI becomes streakier when powder-bed uniformity is reduced.",
    },
}

PRESETS = {
    "reference / stable": ProcessInputs(
        340,
        1250,
        0.12,
        0.06,
        heat_memory=0.35,
        powder_uniformity=0.86,
    ),
    "too cold / lack-of-fusion risk": ProcessInputs(
        238,
        1250,
        0.12,
        0.06,
        heat_memory=0.26,
        powder_uniformity=0.76,
    ),
    "too hot / keyhole-spatter risk": ProcessInputs(
        370,
        1046.38,
        0.12,
        0.06,
        heat_memory=0.72,
        powder_uniformity=0.82,
    ),
    "bad powder spread": ProcessInputs(
        330,
        1280,
        0.12,
        0.06,
        heat_memory=0.38,
        powder_uniformity=0.48,
    ),
}


@dataclass(frozen=True)
class ControlState:
    inputs: ProcessInputs
    second_modality: str
    w_ot: float
    w_second: float
    preset_name: str


@dataclass(frozen=True)
class FeedForwardRecommendation:
    risk_mode: str
    action: str
    current_power_w: float
    recommended_power_w: float
    current_scan_speed_mm_s: float
    recommended_scan_speed_mm_s: float
    delta_power_percent: float
    delta_scan_speed_percent: float
    confidence: float
    rationale: str
    caution: str


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _label(name: str, mode: str = "long") -> str:
    return CLASS_DISPLAY[name][mode]


def _wrap(s: str, width: int = 32) -> str:
    return "\n".join(textwrap.wrap(s, width=width))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if abs(b) < 1e-12:
        return default
    return float(a / b)


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


def _make_seed(inputs: ProcessInputs) -> int:
    return int(
        inputs.laser_power_w
        + inputs.scan_speed_mm_s
        + 1000 * inputs.hatch_distance_mm
        + 1000 * inputs.layer_thickness_mm
    )


# ---------------------------------------------------------------------------
# Physics-informed descriptors
# ---------------------------------------------------------------------------


def compute_physics_features(
    inputs: ProcessInputs,
    *,
    standard_ved_j_mm3: float = STANDARD_VED,
    spot_diameter_mm: float = 0.08,
    thermal_diffusivity_m2_s: float = 6.5e-6,
) -> dict[str, float]:
    """Compute lightweight physics-informed process descriptors.

    These features are not a full thermal simulation. They provide interpretable
    descriptors for physics-informed modelling and ablation studies.
    """

    ved = float(inputs.ved)
    nved = _safe_div(ved, standard_ved_j_mm3, default=np.nan)

    linear_energy = _safe_div(inputs.laser_power_w, inputs.scan_speed_mm_s)
    areal_energy = _safe_div(
        inputs.laser_power_w,
        inputs.scan_speed_mm_s * inputs.hatch_distance_mm,
    )

    residence_time_s = _safe_div(spot_diameter_mm, inputs.scan_speed_mm_s)
    residence_time_ms = 1000.0 * residence_time_s

    diffusion_length_mm = float(
        np.sqrt(max(thermal_diffusivity_m2_s * residence_time_s, 0.0)) * 1000.0
    )

    hatch_to_layer_ratio = _safe_div(
        inputs.hatch_distance_mm,
        inputs.layer_thickness_mm,
    )

    layer_to_hatch_ratio = _safe_div(
        inputs.layer_thickness_mm,
        inputs.hatch_distance_mm,
    )

    heat_accumulation_index = nved * (1.0 + inputs.heat_memory) / max(
        inputs.powder_uniformity,
        1e-6,
    )

    return {
        "ved_j_mm3": float(ved),
        "normalized_ved": float(nved),
        "linear_energy_j_mm": float(linear_energy),
        "areal_energy_j_mm2": float(areal_energy),
        "normalized_laser_power": _safe_div(inputs.laser_power_w, REFERENCE_LASER_POWER_W),
        "normalized_scan_speed": _safe_div(inputs.scan_speed_mm_s, REFERENCE_SCAN_SPEED_MM_S),
        "normalized_hatch_distance": _safe_div(inputs.hatch_distance_mm, REFERENCE_HATCH_DISTANCE_MM),
        "normalized_layer_thickness": _safe_div(inputs.layer_thickness_mm, REFERENCE_LAYER_THICKNESS_MM),
        "hatch_to_layer_ratio": float(hatch_to_layer_ratio),
        "layer_to_hatch_ratio": float(layer_to_hatch_ratio),
        "residence_time_ms": float(residence_time_ms),
        "thermal_diffusion_length_mm": float(diffusion_length_mm),
        "heat_memory": float(inputs.heat_memory),
        "powder_uniformity": float(inputs.powder_uniformity),
        "heat_accumulation_index": float(heat_accumulation_index),
        "low_energy_margin": float(max(0.0, 0.82 - nved)),
        "high_energy_margin": float(max(0.0, nved - 1.18)),
        "distance_from_stable_center": float(abs(nved - 1.0)),
    }


def physics_features_to_frame(features: dict[str, float]) -> pd.DataFrame:
    explanations = {
        "ved_j_mm3": "Volumetric energy density. Useful as a first-order process descriptor.",
        "normalized_ved": "VED divided by the reference VED. Useful for comparing away from the nominal process.",
        "linear_energy_j_mm": "Laser power divided by scan speed. Track-level energy input.",
        "areal_energy_j_mm2": "Power divided by scan speed and hatch distance. Area-normalized input.",
        "normalized_laser_power": "Laser power normalized by the reference value.",
        "normalized_scan_speed": "Scan speed normalized by the reference value.",
        "normalized_hatch_distance": "Hatch distance normalized by the reference value.",
        "normalized_layer_thickness": "Layer thickness normalized by the reference value.",
        "hatch_to_layer_ratio": "Geometric process ratio affecting overlap and fusion stability.",
        "layer_to_hatch_ratio": "Inverse geometric process ratio.",
        "residence_time_ms": "Approximate laser residence time over one spot diameter.",
        "thermal_diffusion_length_mm": "Simple conduction-length proxy during laser residence time.",
        "heat_memory": "Demo estimate of heat carried over from earlier exposure.",
        "powder_uniformity": "Demo estimate of powder-bed spreading quality.",
        "heat_accumulation_index": "Combined normalized VED, heat memory, and powder-condition proxy.",
        "low_energy_margin": "Distance below the lower stable energy boundary.",
        "high_energy_margin": "Distance above the upper stable energy boundary.",
        "distance_from_stable_center": "Absolute distance from normalized VED = 1.",
    }

    rows = []
    for key, value in features.items():
        rows.append(
            {
                "feature": key,
                "value": round(float(value), 5),
                "why it matters": explanations.get(key, ""),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sensor-derived descriptors
# ---------------------------------------------------------------------------


def image_to_gray_array(image_or_path: Any) -> np.ndarray:
    """Convert a PIL image, numpy array, or image path to a grayscale array."""

    if isinstance(image_or_path, (str, Path)):
        img = Image.open(image_or_path).convert("L")
        return np.asarray(img, dtype=np.float32)

    if isinstance(image_or_path, Image.Image):
        return np.asarray(image_or_path.convert("L"), dtype=np.float32)

    if isinstance(image_or_path, np.ndarray):
        arr = image_or_path.astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        return arr

    raise TypeError("image_or_path must be a path, PIL image, or numpy array.")


def compute_sensor_descriptors(image_or_path: Any) -> dict[str, float]:
    """Extract interpretable descriptors from OT, MPM, or PBI images.

    These features are deliberately simple: mode, IQR, variance, hot fraction,
    cold fraction, entropy, texture, and streakiness. They are meant to mimic
    the kind of sensor-summary variables used in data-driven and physics-aware
    PBF-LB studies before real calibrated sensor fields are connected.
    """

    arr = image_to_gray_array(image_or_path)
    flat = arr.ravel()

    if flat.size == 0:
        return {
            "mean_intensity": 0.0,
            "median_intensity": 0.0,
            "mode_intensity": 0.0,
            "variance_intensity": 0.0,
            "std_intensity": 0.0,
            "iqr_intensity": 0.0,
            "p05_intensity": 0.0,
            "p95_intensity": 0.0,
            "p99_intensity": 0.0,
            "hot_pixel_fraction": 0.0,
            "cold_pixel_fraction": 0.0,
            "histogram_entropy": 0.0,
            "texture_energy": 0.0,
            "streakiness": 0.0,
        }

    hist, bin_edges = np.histogram(flat, bins=64, range=(0, 255))
    mode_bin = int(np.argmax(hist))
    mode_intensity = float((bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2)

    p05, p25, p50, p75, p95, p99 = np.percentile(
        flat,
        [5, 25, 50, 75, 95, 99],
    )

    prob = hist.astype(np.float64)
    prob = prob / max(prob.sum(), 1.0)
    entropy = -float(np.sum(prob[prob > 0] * np.log2(prob[prob > 0])))

    if arr.shape[0] > 1:
        grad_y = np.diff(arr, axis=0)
        grad_y_mean = float(np.mean(np.abs(grad_y)))
    else:
        grad_y_mean = 0.0

    if arr.shape[1] > 1:
        grad_x = np.diff(arr, axis=1)
        grad_x_mean = float(np.mean(np.abs(grad_x)))
    else:
        grad_x_mean = 0.0

    texture_energy = grad_x_mean + grad_y_mean
    streakiness = float(np.std(arr.mean(axis=1))) if arr.ndim == 2 else 0.0

    return {
        "mean_intensity": float(np.mean(flat)),
        "median_intensity": float(p50),
        "mode_intensity": float(mode_intensity),
        "variance_intensity": float(np.var(flat)),
        "std_intensity": float(np.std(flat)),
        "iqr_intensity": float(p75 - p25),
        "p05_intensity": float(p05),
        "p95_intensity": float(p95),
        "p99_intensity": float(p99),
        "hot_pixel_fraction": float(np.mean(flat >= p95)),
        "cold_pixel_fraction": float(np.mean(flat <= p05)),
        "histogram_entropy": float(entropy),
        "texture_energy": float(texture_energy),
        "streakiness": float(streakiness),
    }


def sensor_descriptors_to_frame(
    descriptors_by_modality: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for modality, desc in descriptors_by_modality.items():
        for key, value in desc.items():
            rows.append(
                {
                    "modality": modality.upper(),
                    "descriptor": key,
                    "value": round(float(value), 5),
                }
            )
    return pd.DataFrame(rows)


def _current_demo_images(state: ControlState) -> dict[str, Any]:
    seed = _make_seed(state.inputs)
    return {
        "ot": image_from_process(
            state.inputs,
            PATCH_SIZES_HW["ot"],
            "ot",
            seed=seed,
        ),
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


# ---------------------------------------------------------------------------
# Live proxy model and explainability helpers
# ---------------------------------------------------------------------------


def _modality_scores(inputs: ProcessInputs, modality: str) -> dict[str, float]:
    """Transparent sensor proxy for the dashboard.

    It is deliberately simple. Trained repository models still live in the
    training scripts. This function only supports the live dashboard where the
    user changes process settings without loading a checkpoint.
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


def _current_scores(
    state: ControlState,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    ot_scores = _modality_scores(state.inputs, "ot")
    second_scores = _modality_scores(state.inputs, state.second_modality)
    fused = fuse_scores(
        ot_scores,
        second_scores,
        w_a=state.w_ot,
        w_b=state.w_second,
    )
    return ot_scores, second_scores, fused


def _parameter_effect_rows(current: ProcessInputs, reference: ProcessInputs) -> pd.DataFrame:
    rows = []
    items = [
        (
            "laser power P",
            current.laser_power_w,
            reference.laser_power_w,
            "Higher P raises VED and may move the process toward overheating.",
        ),
        (
            "scan speed v",
            current.scan_speed_mm_s,
            reference.scan_speed_mm_s,
            "Higher v lowers VED because the laser spends less time per distance.",
        ),
        (
            "hatch distance h",
            current.hatch_distance_mm,
            reference.hatch_distance_mm,
            "Higher h lowers overlap and lowers VED per volume.",
        ),
        (
            "layer thickness L",
            current.layer_thickness_mm,
            reference.layer_thickness_mm,
            "Higher L spreads the same energy through more material and lowers VED.",
        ),
        (
            "heat memory",
            current.heat_memory,
            reference.heat_memory,
            "Higher heat memory raises high-energy / thermal-accumulation risk.",
        ),
        (
            "powder uniformity",
            current.powder_uniformity,
            reference.powder_uniformity,
            "Lower uniformity raises powder-bed and lack-of-fusion risk.",
        ),
    ]

    for name, value, base, explanation in items:
        change = 0.0 if base == 0 else (value - base) / base * 100.0
        if abs(change) < 1:
            movement = "almost unchanged"
        elif change > 0:
            movement = f"+{change:.1f}%"
        else:
            movement = f"{change:.1f}%"
        rows.append(
            {
                "parameter": name,
                "current": value,
                "change vs preset": movement,
                "meaning": explanation,
            }
        )
    return pd.DataFrame(rows)


def _explanation_rows(
    inputs: ProcessInputs,
    fused: dict[str, float],
    second_modality: str,
) -> pd.DataFrame:
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
                "why it matters": "Accumulated heat can make later layers brighter and less stable even when nominal VED is not extreme.",
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


# ---------------------------------------------------------------------------
# Feed-forward control logic
# ---------------------------------------------------------------------------


def _cap_relative_change(current: float, target: float, max_fraction: float) -> float:
    lower = current * (1.0 - max_fraction)
    upper = current * (1.0 + max_fraction)
    return float(min(max(target, lower), upper))


def recommend_feedforward_control(
    inputs: ProcessInputs,
    fused_scores: dict[str, float],
    sensor_descriptors: dict[str, dict[str, float]] | None = None,
    *,
    standard_ved_j_mm3: float = STANDARD_VED,
    max_power_step_fraction: float = 0.07,
    max_speed_step_fraction: float = 0.07,
) -> FeedForwardRecommendation:
    """Recommend a conservative next-layer correction.

    This is not a machine command. It is an advisory feed-forward decision for
    the next layer or next region after the current in-situ observation.
    """

    features = compute_physics_features(inputs, standard_ved_j_mm3=standard_ved_j_mm3)
    nved = features["normalized_ved"]

    p_stable = float(fused_scores.get("standard", 0.0))
    p_low = float(fused_scores.get("delta_minus_30_ved", 0.0))
    p_high = float(fused_scores.get("delta_plus_30_ved", 0.0))
    risk_score = p_low + p_high

    current_power = float(inputs.laser_power_w)
    current_speed = float(inputs.scan_speed_mm_s)

    target_power = current_power
    target_speed = current_speed
    risk_mode = "stable"
    action = "hold parameters"
    rationale = "The fused output is closest to the stable process window."
    caution = "Continue monitoring; no automatic machine command is issued."

    powder_warning = False
    if sensor_descriptors:
        pbi = sensor_descriptors.get("pbi") or sensor_descriptors.get("PBI")
        if pbi:
            powder_warning = (
                pbi.get("streakiness", 0.0) > 18.0
                or pbi.get("cold_pixel_fraction", 0.0) > 0.08
            )

    if powder_warning and p_low > 0.30:
        risk_mode = "powder-bed / recoating risk"
        action = "inspect powder-bed condition before changing laser parameters"
        rationale = (
            "The low-energy risk is accompanied by powder-bed descriptors. "
            "This suggests that laser correction alone may hide a spreading or recoating problem."
        )
        caution = (
            "Check PBI/recoater condition first. Do not compensate a powder-spreading fault "
            "only by increasing laser power."
        )

    elif p_low > max(p_high, p_stable) or (p_low > 0.40 and nved < 0.92):
        risk_mode = "low-energy / lack-of-fusion risk"
        action = "increase next-layer energy input"
        target_ved = min(standard_ved_j_mm3, inputs.ved * 1.12)

        raw_power_target = target_ved * (
            inputs.scan_speed_mm_s
            * inputs.hatch_distance_mm
            * inputs.layer_thickness_mm
        )
        target_power = _cap_relative_change(
            current_power,
            raw_power_target,
            max_power_step_fraction,
        )

        raw_speed_target = current_speed * (inputs.ved / max(target_ved, 1e-9))
        target_speed = _cap_relative_change(
            current_speed,
            raw_speed_target,
            max_speed_step_fraction,
        )

        rationale = (
            "The fused score is dominated by low-energy risk. "
            "A conservative increase in energy density is recommended for the next layer."
        )
        caution = (
            "Use either power increase or speed reduction, not both at full magnitude, "
            "unless experimentally validated."
        )

    elif p_high > max(p_low, p_stable) or (p_high > 0.40 and nved > 1.08):
        risk_mode = "high-energy / keyhole-spatter risk"
        action = "decrease next-layer energy input"
        target_ved = max(standard_ved_j_mm3, inputs.ved * 0.88)

        raw_power_target = target_ved * (
            inputs.scan_speed_mm_s
            * inputs.hatch_distance_mm
            * inputs.layer_thickness_mm
        )
        target_power = _cap_relative_change(
            current_power,
            raw_power_target,
            max_power_step_fraction,
        )

        raw_speed_target = current_speed * (inputs.ved / max(target_ved, 1e-9))
        target_speed = _cap_relative_change(
            current_speed,
            raw_speed_target,
            max_speed_step_fraction,
        )

        rationale = (
            "The fused score is dominated by high-energy risk. "
            "A conservative reduction in energy density is recommended for the next layer."
        )
        caution = (
            "Reducing power is usually cleaner than increasing speed when scan strategy "
            "and track stability must be preserved."
        )

    confidence = float(min(max(risk_score, 0.0), 1.0))

    return FeedForwardRecommendation(
        risk_mode=risk_mode,
        action=action,
        current_power_w=current_power,
        recommended_power_w=float(target_power),
        current_scan_speed_mm_s=current_speed,
        recommended_scan_speed_mm_s=float(target_speed),
        delta_power_percent=100.0 * (target_power - current_power) / max(current_power, 1e-9),
        delta_scan_speed_percent=100.0 * (target_speed - current_speed) / max(current_speed, 1e-9),
        confidence=confidence,
        rationale=rationale,
        caution=caution,
    )


def recommendation_to_frame(rec: FeedForwardRecommendation) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item": "risk mode", "value": rec.risk_mode},
            {"item": "recommended action", "value": rec.action},
            {"item": "current laser power", "value": f"{rec.current_power_w:.1f} W"},
            {"item": "recommended laser power", "value": f"{rec.recommended_power_w:.1f} W"},
            {"item": "power change", "value": f"{rec.delta_power_percent:+.2f}%"},
            {"item": "current scan speed", "value": f"{rec.current_scan_speed_mm_s:.1f} mm/s"},
            {"item": "recommended scan speed", "value": f"{rec.recommended_scan_speed_mm_s:.1f} mm/s"},
            {"item": "scan-speed change", "value": f"{rec.delta_scan_speed_percent:+.2f}%"},
            {"item": "confidence", "value": f"{rec.confidence:.2f}"},
            {"item": "rationale", "value": rec.rationale},
            {"item": "caution", "value": rec.caution},
        ]
    )


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


def _plot_sensitivity(
    inputs: ProcessInputs,
    second_modality: str,
    w_ot: float,
    w_second: float,
) -> plt.Figure:
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


def _plot_layer_story(
    start: int,
    n_layers: int,
    mode: str,
    healing_capacity: int,
) -> tuple[plt.Figure, pd.DataFrame, str]:
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
        msg = (
            f"Disturbance depth is {n_layers} layers, above the selected healing capacity of "
            f"{healing_capacity}. Residual risk is kept after normal exposure resumes."
        )
    else:
        msg = (
            f"Disturbance depth is {n_layers} layers, within the selected healing capacity of "
            f"{healing_capacity}. The signal decays after normal exposure resumes."
        )

    return fig, df, msg


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------


def _sidebar_controls() -> ControlState:
    st.sidebar.title("Controls")
    st.sidebar.caption(
        "Change these values and watch VED, sensor scores, physics descriptors, "
        "and the feed-forward advisory update immediately."
    )

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
        second_modality = st.radio(
            "Second channel",
            ["mpm", "pbi"],
            format_func=lambda x: x.upper(),
            horizontal=True,
        )
        w_ot = st.slider("OT weight", 0.0, 1.0, 0.50, 0.05)
        w_second = 1.0 - w_ot
        st.caption(f"Second-channel weight: {w_second:.2f}")

    return ControlState(
        inputs=ProcessInputs(
            laser_power,
            scan_speed,
            hatch,
            layer,
            heat_memory,
            powder_uniformity,
        ),
        second_modality=second_modality,
        w_ot=w_ot,
        w_second=w_second,
        preset_name=preset_name,
    )


def _show_system_summary() -> None:
    st.title("LayerWise-QC dashboard")
    st.markdown(
        """
This page explains a physics-informed layer-wise quality-monitoring workflow for laser powder-bed fusion. It is not only an image viewer: each change in the controls updates the energy-density calculation, sensor-channel scores, physics-informed descriptors, the fused decision, and the feed-forward advisory.

**Reading the page:** low VED usually means lack-of-fusion risk; high VED usually means keyhole/spatter risk; poor powder spreading mainly affects the powder-bed imaging path. The current app remains a prototype until real in-situ images, machine metadata, and experimental ground truth are connected.
        """.strip()
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.info("1. Set process parameters")
    c2.info("2. Inspect sensor proxies")
    c3.info("3. Extract physics features")
    c4.info("4. Recommend next-layer action")


def _show_decision_panel(
    state: ControlState,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    inputs = state.inputs
    ot_scores, second_scores, fused = _current_scores(state)
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
            [
                _bar_df(ot_scores, "ot"),
                _bar_df(second_scores, state.second_modality),
                _bar_df(fused, "fused"),
            ],
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

Increasing power raises energy density. Increasing scan speed, hatch distance, or layer thickness lowers it. Heat memory and powder uniformity are extra process-state sliders used to show how sensor channels can disagree even when VED is similar.
        """.strip()
    )

    ref = PRESETS[state.preset_name]
    a, b = st.columns([0.50, 0.50], gap="large")
    with a:
        st.subheader("Parameter movement")
        st.dataframe(
            _parameter_effect_rows(state.inputs, ref),
            hide_index=True,
            use_container_width=True,
        )
    with b:
        st.subheader("Decision reasons")
        st.dataframe(
            _explanation_rows(state.inputs, fused, state.second_modality),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Sensitivity check")
    st.caption(
        "This plot keeps all sliders fixed except scan speed. It shows why scan speed "
        "is a strong lever: faster scanning lowers VED, slower scanning raises VED."
    )
    st.pyplot(
        _plot_sensitivity(state.inputs, state.second_modality, state.w_ot, state.w_second),
        use_container_width=True,
    )


def _show_physics_informed_panel(state: ControlState) -> None:
    st.header("Physics-informed feature space")
    st.markdown(
        """
This panel elevates the dashboard from a VED-only demonstration into a physics-informed feature prototype. The goal is to compare process-only descriptors, sensor-derived descriptors, and fused descriptors before real model training.
        """.strip()
    )

    features = compute_physics_features(state.inputs)

    st.subheader("Process-derived physics descriptors")
    st.dataframe(
        physics_features_to_frame(features),
        hide_index=True,
        use_container_width=True,
    )

    descriptors = _current_sensor_descriptors(state)

    st.subheader("Sensor-derived descriptors")
    st.caption(
        "These are descriptors from the current OT and second-channel demo patches. "
        "For real experiments, the same functions can be applied to actual OT, MPM, or PBI images."
    )
    st.dataframe(
        sensor_descriptors_to_frame(descriptors),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Ablation-ready modelling logic")
    st.markdown(
        """
A serious version of this prototype should compare the following baselines:

1. **process-only model:** laser power, scan speed, hatch distance, layer thickness, VED, normalized VED;
2. **sensor-descriptor model:** thermal/image descriptors such as mode, IQR, variance, texture, and streakiness;
3. **image model:** CNN or vision backbone using OT / MPM / PBI patches directly;
4. **hybrid physics-informed fusion:** process descriptors + sensor descriptors + image-model scores.
        """.strip()
    )

    st.info(
        "This is the key research upgrade: the app now exposes the variables needed for "
        "process-only, sensor-only, image-only, and hybrid physics-informed ablation studies."
    )


def _show_feedforward_panel(state: ControlState) -> None:
    st.header("Feed-forward control advisory")
    st.markdown(
        """
This panel converts monitoring into a next-layer advisory decision. It does not claim direct machine control. It recommends conservative parameter movement for the next layer based on the fused risk state and sensor descriptors.
        """.strip()
    )

    _, _, fused = _current_scores(state)
    descriptors = _current_sensor_descriptors(state)

    rec = recommend_feedforward_control(
        state.inputs,
        fused,
        sensor_descriptors=descriptors,
    )

    a, b, c = st.columns(3)
    a.metric("Risk mode", rec.risk_mode)
    b.metric("Power change", f"{rec.delta_power_percent:+.2f}%")
    c.metric("Speed change", f"{rec.delta_scan_speed_percent:+.2f}%")

    st.dataframe(
        recommendation_to_frame(rec),
        hide_index=True,
        use_container_width=True,
    )

    st.warning(
        "This is an advisory controller, not a machine command. Real deployment would "
        "require machine-specific calibration, safety limits, controller permissions, "
        "and validation against real density, porosity, metallography, or CT data."
    )

    st.subheader("Control logic")
    st.code(
        """
if low-energy / lack-of-fusion risk dominates:
    increase laser power slightly
    or reduce scan speed slightly

if high-energy / keyhole-spatter risk dominates:
    reduce laser power slightly
    or increase scan speed slightly

if powder-bed descriptors look abnormal:
    flag recoating / powder-spreading issue
    do not compensate only by laser power
        """.strip(),
        language="text",
    )


def _show_sensor_panel(
    state: ControlState,
    ot_scores: dict[str, float],
    second_scores: dict[str, float],
    fused: dict[str, float],
) -> None:
    st.header("Sensor view")
    st.write(
        "The images below are generated demo patches. They are not lab measurements. "
        "Their job is to make the data path understandable before real OT, MPM, or PBI files are connected."
    )

    images = _current_demo_images(state)

    cols = st.columns([0.31, 0.31, 0.38], gap="large")
    with cols[0]:
        st.subheader("OT patch")
        st.image(images["ot"], use_container_width=True)
        st.caption(SENSOR_TEXT["ot"]["what"])
        st.write(f"OT prediction: **{_label(_prediction(ot_scores))}**")

    with cols[1]:
        st.subheader(f"{state.second_modality.upper()} patch")
        st.image(images[state.second_modality], use_container_width=True)
        st.caption(SENSOR_TEXT[state.second_modality]["what"])
        st.write(f"{state.second_modality.upper()} prediction: **{_label(_prediction(second_scores))}**")

    with cols[2]:
        st.subheader("Fusion table")
        table = pd.concat(
            [
                _bar_df(ot_scores, "ot"),
                _bar_df(second_scores, state.second_modality),
                _bar_df(fused, "fused"),
            ],
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
            rows.append(
                {
                    "channel": key.upper(),
                    "role": SENSOR_TEXT[key]["what"],
                    "demo behavior": SENSOR_TEXT[key]["demo"],
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _show_sample_library() -> None:
    st.header("Built-in sample library")
    manifest = _read_manifest()
    if manifest is None or manifest.empty:
        st.info(
            "No demo sample manifest found. Run `python scripts/make_synthetic_dataset.py --out data/demo_samples --layers 18`."
        )
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
    st.header("Layer history / healing sketch")
    st.write(
        "This part is a simple timeline sketch. It shows why layer-wise monitoring matters: "
        "a short disturbed region may disappear after standard exposure, while a deeper disturbed stack can leave residual risk."
    )

    a, b, c, d = st.columns(4)
    with a:
        n_layers = st.slider("Disturbed layers", 1, 12, 7)
    with b:
        start = st.slider("Disturbance starts at layer", 5, 70, 20)
    with c:
        healing_capacity = st.slider("Healing capacity assumption", 1, 10, 7)
    with d:
        mode = st.radio(
            "Disturbance type",
            ["delta_minus_30_ved", "delta_plus_30_ved"],
            format_func=lambda x: _label(x, "short"),
        )

    fig, _, msg = _plot_layer_story(start, n_layers, mode, healing_capacity)
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
    st.header("How this connects to the training code")
    st.markdown(
        """
The live dashboard uses transparent proxy scores so it can run without trained checkpoints. The repository also contains the actual training path:

1. prepare a CSV manifest with one row per layer or patch,
2. train one model for OT,
3. train one model for the second channel, either MPM or PBI,
4. fuse output probabilities,
5. inspect metrics and Grad-CAM overlays.

Minimum real-data manifest columns:

```text
sample_id,class_idx,class_name,ot_path,mpm_path,pbi_path,laser_power_w,scan_speed_mm_s,hatch_distance_mm,layer_thickness_mm,ved_j_mm3
```

Recommended research-grade manifest columns:

```text
sample_id,build_id,specimen_id,layer,region_id,class_idx,class_name,
ot_path,mpm_path,pbi_path,
laser_power_w,scan_speed_mm_s,hatch_distance_mm,layer_thickness_mm,ved_j_mm3,
relative_density,porosity_fraction,ct_label,metallography_label,
normalized_ved,normalized_laser_power,thermal_mode,thermal_iqr,thermal_variance,powder_streakiness
```

For a serious experiment, the dashboard should be fed by actual layer-wise image patches and trained checkpoints, not only by the proxy score function used here.
        """.strip()
    )

    st.subheader("Current prototype data flow")
    st.code(
        """process settings + layer image patches
        ↓
OT proxy / OT model                 second-channel proxy / MPM or PBI model
        ↓                                      ↓
OT class scores                       second-channel class scores
        \                                      /
         late fusion of probabilities
                  ↓
        layer-wise quality state + explanation
                  ↓
        feed-forward advisory for the next layer""",
        language="text",
    )

    st.subheader("Research upgrade path")
    st.markdown(
        """
The next code layer should export a feature table from real manifests:

```text
scripts/build_feature_table.py
```

That script should compute one row per layer or region with process parameters, physics-informed descriptors, sensor-image descriptors, model scores, and experimental ground truth. Then the project can compare:

1. process-only model,
2. sensor-descriptor-only model,
3. image-CNN-only model,
4. hybrid physics-informed sensor-fusion model,
5. feed-forward advisory policy.
        """.strip()
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="LayerWise-QC",
        page_icon="🧩",
        layout="wide",
    )

    state = _sidebar_controls()
    _show_system_summary()

    tabs = st.tabs(
        [
            "live decision",
            "physics-informed features",
            "feed-forward control",
            "sensor view",
            "sample library",
            "layer history",
            "pipeline",
        ]
    )

    with tabs[0]:
        _, _, fused = _show_decision_panel(state)
        _show_why_panel(state, fused)

    with tabs[1]:
        _show_physics_informed_panel(state)

    with tabs[2]:
        _show_feedforward_panel(state)

    with tabs[3]:
        ot_scores, second_scores, fused = _current_scores(state)
        _show_sensor_panel(state, ot_scores, second_scores, fused)

    with tabs[4]:
        _show_sample_library()

    with tabs[5]:
        _show_layer_story()

    with tabs[6]:
        _show_pipeline_panel()


if __name__ == "__main__":
    main()
