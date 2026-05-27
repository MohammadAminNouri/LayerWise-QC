"""Human-readable explanation helpers for LayerWise-QC."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .fusion_analysis import label, prediction
from .physics_features import STANDARD_VED_J_MM3
from .simulation import ProcessInputs


def explain_process_inputs(inputs: ProcessInputs) -> pd.DataFrame:
    """Explain the role of each process input/slider."""
    rows = [
        {
            "input": "laser power P",
            "current value": f"{inputs.laser_power_w:.1f} W",
            "meaning": "Higher power increases energy input. Too high can increase keyhole/spatter risk.",
        },
        {
            "input": "scan speed v",
            "current value": f"{inputs.scan_speed_mm_s:.1f} mm/s",
            "meaning": "Higher speed lowers energy per unit length. Too high can increase lack-of-fusion risk.",
        },
        {
            "input": "hatch distance h",
            "current value": f"{inputs.hatch_distance_mm:.3f} mm",
            "meaning": "Larger hatch distance reduces track overlap and lowers energy per volume.",
        },
        {
            "input": "layer thickness t",
            "current value": f"{inputs.layer_thickness_mm:.3f} mm",
            "meaning": "Thicker layers need more energy to melt completely.",
        },
        {
            "input": "heat memory",
            "current value": f"{inputs.heat_memory:.2f}",
            "meaning": "Demo proxy for heat accumulation from previous layers.",
        },
        {
            "input": "powder-bed uniformity",
            "current value": f"{inputs.powder_uniformity:.2f}",
            "meaning": "Demo proxy for powder spreading quality or recoater-related problems.",
        },
    ]
    return pd.DataFrame(rows)


def explain_decision(
    inputs: ProcessInputs,
    fused_scores: Dict[str, float],
    sensor_descriptors: Dict[str, Dict[str, float]] | None = None,
    *,
    second_modality: str = "mpm",
) -> List[str]:
    """Convert numeric state into readable reasons."""
    reasons: List[str] = []
    ratio = float(inputs.ved / STANDARD_VED_J_MM3)

    if ratio < 0.82:
        reasons.append(
            f"VED is {ratio:.2f}× the reference value, below the stable window, so lack-of-fusion risk increases."
        )
    elif ratio > 1.18:
        reasons.append(
            f"VED is {ratio:.2f}× the reference value, above the stable window, so keyhole/spatter risk increases."
        )
    else:
        reasons.append(
            f"VED is {ratio:.2f}× the reference value, inside the demo stable window."
        )

    if inputs.heat_memory > 0.65:
        reasons.append(
            "Heat memory is high, so thermal accumulation may make the current layer hotter than the nominal VED suggests."
        )

    if inputs.powder_uniformity < 0.70:
        reasons.append(
            "Powder-bed uniformity is low, so powder spreading or recoater-related defects are possible."
        )

    if sensor_descriptors:
        pbi = sensor_descriptors.get("pbi") or sensor_descriptors.get("PBI")
        if pbi and pbi.get("streakiness", 0.0) > 18.0:
            reasons.append(
                "PBI streakiness is high, which supports a powder-bed or recoating anomaly interpretation."
            )

        active = sensor_descriptors.get(second_modality)
        if active and active.get("iqr_intensity", 0.0) > 55.0:
            reasons.append(
                f"{second_modality.upper()} intensity spread is high, so the second sensor channel is not uniform."
            )

    pred = prediction(fused_scores)
    reasons.append(f"The final fused output is: {label(pred)}.")

    return reasons


def explanation_frame(reasons: List[str]) -> pd.DataFrame:
    """Convert reasons list into a table."""
    return pd.DataFrame(
        [{"step": i + 1, "explanation": reason} for i, reason in enumerate(reasons)]
    )


def overview_outputs_frame() -> pd.DataFrame:
    """Explain what the app gives to the user."""
    return pd.DataFrame(
        [
            {
                "output": "Layer quality class",
                "meaning": "Stable, low-energy/lack-of-fusion risk, or high-energy/keyhole-spatter risk.",
            },
            {
                "output": "Risk index",
                "meaning": "Combined probability-like score of unstable classes.",
            },
            {
                "output": "Explanation",
                "meaning": "Human-readable reasons for the decision.",
            },
            {
                "output": "Sensor agreement",
                "meaning": "Whether OT and MPM/PBI agree or conflict.",
            },
            {
                "output": "Physics-informed features",
                "meaning": "Interpretable process descriptors for research and ablation.",
            },
            {
                "output": "Feed-forward advisory",
                "meaning": "Conservative next-layer suggestion, not an automatic machine command.",
            },
            {
                "output": "Data readiness",
                "meaning": "Checks whether real data and ground truth are available.",
            },
        ]
    )
