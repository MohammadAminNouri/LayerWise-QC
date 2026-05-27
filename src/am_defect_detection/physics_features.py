"""Physics-informed and sensor-derived feature extraction for LayerWise-QC.

These features are designed to make the prototype closer to the literature on
PBF-LB in-situ monitoring and physics-informed machine learning.

The functions here are intentionally lightweight:
- no heavy dependencies,
- usable inside Streamlit,
- usable later in training scripts,
- works with synthetic demo images now and real OT/MPM/PBI images later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from PIL import Image

from .simulation import ProcessInputs


STANDARD_VED_J_MM3 = 37.78
REFERENCE_LASER_POWER_W = 340.0
REFERENCE_SCAN_SPEED_MM_S = 1250.0
REFERENCE_HATCH_DISTANCE_MM = 0.12
REFERENCE_LAYER_THICKNESS_MM = 0.06


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if abs(b) < 1e-12:
        return default
    return float(a / b)


def compute_physics_features(
    inputs: ProcessInputs,
    *,
    standard_ved_j_mm3: float = STANDARD_VED_J_MM3,
    spot_diameter_mm: float = 0.08,
    thermal_diffusivity_m2_s: float = 6.5e-6,
) -> Dict[str, float]:
    """Compute physics-informed process descriptors.

    Parameters
    ----------
    inputs:
        ProcessInputs from the current dashboard or manifest.
    standard_ved_j_mm3:
        Reference VED used for normalization.
    spot_diameter_mm:
        Approximate laser spot diameter. Used only for residence-time proxy.
    thermal_diffusivity_m2_s:
        Approximate thermal diffusivity. For 316L, ESAFORM uses around
        6.5e-6 m^2/s as a standard-property estimate.

    Returns
    -------
    dict
        A flat feature dictionary suitable for Streamlit tables, CSV export,
        or later ML baselines.
    """

    ved = inputs.ved
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
        "normalized_laser_power": _safe_div(
            inputs.laser_power_w,
            REFERENCE_LASER_POWER_W,
        ),
        "normalized_scan_speed": _safe_div(
            inputs.scan_speed_mm_s,
            REFERENCE_SCAN_SPEED_MM_S,
        ),
        "normalized_hatch_distance": _safe_div(
            inputs.hatch_distance_mm,
            REFERENCE_HATCH_DISTANCE_MM,
        ),
        "normalized_layer_thickness": _safe_div(
            inputs.layer_thickness_mm,
            REFERENCE_LAYER_THICKNESS_MM,
        ),
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


def physics_features_to_frame(features: Dict[str, float]) -> pd.DataFrame:
    """Return a clean dataframe for the Streamlit app."""

    explanations = {
        "ved_j_mm3": "Volumetric energy density; useful first-order process descriptor.",
        "normalized_ved": "VED divided by reference VED. Used as a transferable normalized descriptor.",
        "linear_energy_j_mm": "Laser power divided by scan speed; track-level energy input.",
        "areal_energy_j_mm2": "Power divided by scan speed and hatch distance.",
        "normalized_laser_power": "Laser power normalized by reference laser power.",
        "normalized_scan_speed": "Scan speed normalized by reference scan speed.",
        "normalized_hatch_distance": "Hatch distance normalized by reference hatch distance.",
        "normalized_layer_thickness": "Layer thickness normalized by reference layer thickness.",
        "hatch_to_layer_ratio": "Geometric process ratio affecting overlap and fusion stability.",
        "layer_to_hatch_ratio": "Inverse geometric process ratio.",
        "residence_time_ms": "Approximate laser residence time over one spot diameter.",
        "thermal_diffusion_length_mm": "Simple conduction-length proxy during laser residence time.",
        "heat_memory": "User/demo estimate of prior heat accumulation.",
        "powder_uniformity": "User/demo estimate of powder-bed spreading condition.",
        "heat_accumulation_index": "Combined normalized VED, heat memory, and powder condition proxy.",
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


def image_to_gray_array(image_or_path: Any) -> np.ndarray:
    """Convert PIL image or file path to grayscale float array in [0, 255]."""

    if isinstance(image_or_path, (str, Path)):
        img = Image.open(image_or_path).convert("L")
    elif isinstance(image_or_path, Image.Image):
        img = image_or_path.convert("L")
    else:
        raise TypeError("image_or_path must be a PIL.Image.Image or path-like object.")

    return np.asarray(img, dtype=np.float32)


def compute_sensor_descriptors(image_or_path: Any) -> Dict[str, float]:
    """Extract lightweight image descriptors from OT, MPM, or PBI images.

    These descriptors are deliberately simple and interpretable. They are not
    a replacement for calibrated temperature fields, but they give the prototype
    a path toward the literature-style thermal/image features:
    mode, IQR, variance, hot fraction, cold fraction, texture, and streakiness.
    """

    arr = image_to_gray_array(image_or_path)
    flat = arr.ravel()

    hist, bin_edges = np.histogram(flat, bins=64, range=(0, 255))
    mode_bin = int(np.argmax(hist))
    mode_intensity = float((bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2)

    p05, p25, p50, p75, p95, p99 = np.percentile(
        flat,
        [5, 25, 50, 75, 95, 99],
    )

    # Entropy-like histogram descriptor.
    prob = hist.astype(np.float64)
    prob = prob / max(prob.sum(), 1.0)
    entropy = -float(np.sum(prob[prob > 0] * np.log2(prob[prob > 0])))

    # Simple spatial descriptors.
    grad_y = np.diff(arr, axis=0)
    grad_x = np.diff(arr, axis=1)
    texture_energy = float(np.mean(np.abs(grad_x)) + np.mean(np.abs(grad_y)))

    # PBI-like streak descriptor: strong row-to-row variation.
    row_mean = arr.mean(axis=1)
    streakiness = float(np.std(row_mean))

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
    descriptors_by_modality: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Convert nested sensor descriptors to a Streamlit-friendly dataframe."""

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
