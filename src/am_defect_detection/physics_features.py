"""Physics-informed process descriptors for LayerWise-QC.

These functions turn machine settings into interpretable descriptors that can be
used in the app, in feature-table exports, and later in ML baselines.

The goal is not to replace a full thermal model. The goal is to make the current
dashboard more research-facing by exposing physically meaningful quantities such
as normalized VED, linear energy, areal energy, heat-memory proxies, and simple
residence-time / diffusion-length proxies.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .constants import REFERENCE_SPOT_SIZE_UM, calculate_beam_area_mm2, spot_size_um_to_mm
from .simulation import ProcessInputs


STANDARD_VED_J_MM3 = 37.78

REFERENCE_LASER_POWER_W = 340.0
REFERENCE_SCAN_SPEED_MM_S = 1250.0
REFERENCE_HATCH_DISTANCE_MM = 0.12
REFERENCE_LAYER_THICKNESS_MM = 0.06
REFERENCE_SPOT_DIAMETER_MM = REFERENCE_SPOT_SIZE_UM / 1000.0


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Divide safely and return a default if the denominator is almost zero."""
    if abs(float(b)) < 1e-12:
        return float(default)
    return float(a) / float(b)


def compute_physics_features(
    inputs: ProcessInputs,
    *,
    standard_ved_j_mm3: float = STANDARD_VED_J_MM3,
    spot_diameter_mm: float | None = None,
    thermal_diffusivity_m2_s: float = 6.5e-6,
) -> Dict[str, float]:
    """Compute physics-informed descriptors from process inputs.

    Parameters
    ----------
    inputs:
        ProcessInputs used by the app.
    standard_ved_j_mm3:
        Reference VED for normalization.
    spot_diameter_mm:
        Approximate laser spot diameter used for a residence-time proxy.
    thermal_diffusivity_m2_s:
        Approximate thermal diffusivity used for a simple heat-spreading proxy.

    Returns
    -------
    dict
        Flat dictionary of numeric features.
    """

    ved = float(inputs.ved)
    normalized_ved = safe_div(ved, standard_ved_j_mm3, default=np.nan)

    linear_energy_j_mm = safe_div(inputs.laser_power_w, inputs.scan_speed_mm_s)
    areal_energy_j_mm2 = safe_div(
        inputs.laser_power_w,
        inputs.scan_speed_mm_s * inputs.hatch_distance_mm,
    )

    if spot_diameter_mm is None:
        spot_size_um = float(getattr(inputs, "spot_size_um", REFERENCE_SPOT_SIZE_UM))
        spot_diameter_mm = spot_size_um_to_mm(spot_size_um)
    else:
        spot_diameter_mm = float(spot_diameter_mm)
        spot_size_um = spot_diameter_mm * 1000.0

    beam_radius_mm = spot_diameter_mm / 2.0
    beam_area_mm2 = calculate_beam_area_mm2(spot_size_um)
    power_density_w_mm2 = safe_div(inputs.laser_power_w, beam_area_mm2, default=np.nan)
    normalized_spot_size = safe_div(spot_size_um, REFERENCE_SPOT_SIZE_UM, default=np.nan)
    reference_beam_area_mm2 = calculate_beam_area_mm2(REFERENCE_SPOT_SIZE_UM)
    reference_power_density = safe_div(REFERENCE_LASER_POWER_W, reference_beam_area_mm2, default=np.nan)
    normalized_power_density = safe_div(power_density_w_mm2, reference_power_density, default=np.nan)
    spot_overlap_ratio = safe_div(spot_diameter_mm, inputs.hatch_distance_mm, default=np.nan)
    hatch_to_spot_ratio = safe_div(inputs.hatch_distance_mm, spot_diameter_mm, default=np.nan)
    spot_to_layer_ratio = safe_div(spot_diameter_mm, inputs.layer_thickness_mm, default=np.nan)

    residence_time_s = safe_div(spot_diameter_mm, inputs.scan_speed_mm_s)
    residence_time_ms = 1000.0 * residence_time_s

    thermal_diffusion_length_mm = float(
        np.sqrt(max(thermal_diffusivity_m2_s * residence_time_s, 0.0)) * 1000.0
    )

    hatch_to_layer_ratio = safe_div(
        inputs.hatch_distance_mm,
        inputs.layer_thickness_mm,
    )
    layer_to_hatch_ratio = safe_div(
        inputs.layer_thickness_mm,
        inputs.hatch_distance_mm,
    )

    # This is a transparent dashboard proxy, not a validated physical law.
    heat_accumulation_index = normalized_ved * (1.0 + inputs.heat_memory) / max(
        inputs.powder_uniformity,
        1e-6,
    )

    return {
        "ved_j_mm3": float(ved),
        "normalized_ved": float(normalized_ved),
        "linear_energy_j_mm": float(linear_energy_j_mm),
        "areal_energy_j_mm2": float(areal_energy_j_mm2),
        "normalized_laser_power": safe_div(inputs.laser_power_w, REFERENCE_LASER_POWER_W),
        "normalized_scan_speed": safe_div(inputs.scan_speed_mm_s, REFERENCE_SCAN_SPEED_MM_S),
        "normalized_hatch_distance": safe_div(inputs.hatch_distance_mm, REFERENCE_HATCH_DISTANCE_MM),
        "normalized_layer_thickness": safe_div(inputs.layer_thickness_mm, REFERENCE_LAYER_THICKNESS_MM),
        "spot_size_um": float(spot_size_um),
        "spot_size_mm": float(spot_diameter_mm),
        "beam_radius_mm": float(beam_radius_mm),
        "beam_area_mm2": float(beam_area_mm2),
        "power_density_w_mm2": float(power_density_w_mm2),
        "normalized_spot_size": float(normalized_spot_size),
        "normalized_power_density": float(normalized_power_density),
        "spot_overlap_ratio": float(spot_overlap_ratio),
        "hatch_to_spot_ratio": float(hatch_to_spot_ratio),
        "spot_to_layer_ratio": float(spot_to_layer_ratio),
        "hatch_to_layer_ratio": float(hatch_to_layer_ratio),
        "layer_to_hatch_ratio": float(layer_to_hatch_ratio),
        "residence_time_ms": float(residence_time_ms),
        "thermal_diffusion_length_mm": float(thermal_diffusion_length_mm),
        "heat_memory": float(inputs.heat_memory),
        "powder_uniformity": float(inputs.powder_uniformity),
        "heat_accumulation_index": float(heat_accumulation_index),
        "power_density_x_residence_proxy": float(power_density_w_mm2 * residence_time_ms),
        "ved_x_power_density": float(ved * power_density_w_mm2),
        "low_energy_margin": float(max(0.0, 0.82 - normalized_ved)),
        "high_energy_margin": float(max(0.0, normalized_ved - 1.18)),
        "distance_from_stable_center": float(abs(normalized_ved - 1.0)),
    }


def physics_feature_explanations() -> Dict[str, str]:
    """Human-readable meaning of each physics feature."""
    return {
        "ved_j_mm3": "Volumetric energy density. First-order energy input per unit volume.",
        "normalized_ved": "Current VED divided by the reference VED.",
        "linear_energy_j_mm": "Laser power divided by scan speed. Approximate track-level energy input.",
        "areal_energy_j_mm2": "Laser power divided by scan speed and hatch distance.",
        "normalized_laser_power": "Laser power normalized by the reference laser power.",
        "normalized_scan_speed": "Scan speed normalized by the reference scan speed.",
        "normalized_hatch_distance": "Hatch distance normalized by the reference hatch distance.",
        "normalized_layer_thickness": "Layer thickness normalized by the reference layer thickness.",
        "spot_size_um": "Laser spot size / beam diameter in micrometres.",
        "spot_size_mm": "Laser spot size / beam diameter in millimetres.",
        "beam_radius_mm": "Nominal beam radius used for beam-area calculation.",
        "beam_area_mm2": "Nominal circular beam area. Smaller area concentrates energy more strongly.",
        "power_density_w_mm2": "Laser power divided by nominal beam area; captures energy concentration ignored by VED.",
        "normalized_spot_size": "Spot size divided by the reference spot size.",
        "normalized_power_density": "Power density divided by the reference power density.",
        "spot_overlap_ratio": "Spot diameter divided by hatch distance; a simple track-overlap proxy.",
        "hatch_to_spot_ratio": "Hatch distance divided by spot diameter; high values can indicate weak overlap.",
        "spot_to_layer_ratio": "Spot diameter divided by layer thickness.",
        "hatch_to_layer_ratio": "Geometric ratio related to scan-track overlap and layer geometry.",
        "layer_to_hatch_ratio": "Inverse geometry ratio.",
        "residence_time_ms": "Approximate laser interaction time over one spot diameter.",
        "thermal_diffusion_length_mm": "Simple heat-spreading proxy during laser residence time.",
        "heat_memory": "Demo estimate of heat accumulation from previous exposure.",
        "powder_uniformity": "Demo estimate of powder spreading quality.",
        "heat_accumulation_index": "Proxy combining normalized VED, heat memory, and powder condition.",
        "power_density_x_residence_proxy": "Interaction proxy combining beam concentration and residence time.",
        "ved_x_power_density": "Interaction proxy combining volumetric energy and beam concentration.",
        "low_energy_margin": "Distance below the lower stable-window boundary.",
        "high_energy_margin": "Distance above the upper stable-window boundary.",
        "distance_from_stable_center": "Absolute distance from normalized VED = 1.",
    }


def physics_features_to_frame(features: Dict[str, float]) -> pd.DataFrame:
    """Convert physics features into a Streamlit-friendly table."""
    explanations = physics_feature_explanations()
    return pd.DataFrame(
        [
            {
                "feature": key,
                "value": round(float(value), 5),
                "why it matters": explanations.get(key, ""),
            }
            for key, value in features.items()
        ]
    )
