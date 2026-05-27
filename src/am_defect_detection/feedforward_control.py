"""Layer-to-layer feed-forward advisory control for LayerWise-QC.

The functions in this file do not control a machine. They produce conservative,
human-readable recommendations for the next layer based on the current fused risk
state and sensor descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from .physics_features import STANDARD_VED_J_MM3, compute_physics_features
from .simulation import ProcessInputs


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


def cap_relative_change(current: float, target: float, max_fraction: float) -> float:
    """Limit a recommendation to a conservative relative step."""
    lower = current * (1.0 - max_fraction)
    upper = current * (1.0 + max_fraction)
    return float(min(max(target, lower), upper))


def recommend_feedforward_control(
    inputs: ProcessInputs,
    fused_scores: Dict[str, float],
    sensor_descriptors: Optional[Dict[str, Dict[str, float]]] = None,
    *,
    standard_ved_j_mm3: float = STANDARD_VED_J_MM3,
    max_power_step_fraction: float = 0.07,
    max_speed_step_fraction: float = 0.07,
) -> FeedForwardRecommendation:
    """Recommend a conservative next-layer correction."""

    features = compute_physics_features(inputs, standard_ved_j_mm3=standard_ved_j_mm3)
    normalized_ved = features["normalized_ved"]

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
    caution = "Continue monitoring. No automatic machine command is issued."

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

    elif p_low > max(p_high, p_stable) or (p_low > 0.40 and normalized_ved < 0.92):
        risk_mode = "low-energy / lack-of-fusion risk"
        action = "increase next-layer energy input"

        target_ved = min(standard_ved_j_mm3, inputs.ved * 1.12)

        raw_power_target = target_ved * (
            inputs.scan_speed_mm_s
            * inputs.hatch_distance_mm
            * inputs.layer_thickness_mm
        )
        target_power = cap_relative_change(
            current_power,
            raw_power_target,
            max_power_step_fraction,
        )

        raw_speed_target = current_speed * (inputs.ved / max(target_ved, 1e-9))
        target_speed = cap_relative_change(
            current_speed,
            raw_speed_target,
            max_speed_step_fraction,
        )

        rationale = (
            "The fused score is dominated by low-energy risk. "
            "A conservative increase in energy density is recommended for the next layer."
        )
        caution = (
            "Use either the power increase or the scan-speed reduction, not both at full magnitude, "
            "unless experimentally validated."
        )

    elif p_high > max(p_low, p_stable) or (p_high > 0.40 and normalized_ved > 1.08):
        risk_mode = "high-energy / keyhole-spatter risk"
        action = "decrease next-layer energy input"

        target_ved = max(standard_ved_j_mm3, inputs.ved * 0.88)

        raw_power_target = target_ved * (
            inputs.scan_speed_mm_s
            * inputs.hatch_distance_mm
            * inputs.layer_thickness_mm
        )
        target_power = cap_relative_change(
            current_power,
            raw_power_target,
            max_power_step_fraction,
        )

        raw_speed_target = current_speed * (inputs.ved / max(target_ved, 1e-9))
        target_speed = cap_relative_change(
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
    """Convert a feed-forward recommendation to a display table."""
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
