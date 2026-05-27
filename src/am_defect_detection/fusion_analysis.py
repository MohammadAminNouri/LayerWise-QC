"""Fusion and uncertainty utilities for LayerWise-QC."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from .constants import CLASS_NAMES


CLASS_LONG = {
    "standard": "stable process window",
    "delta_minus_30_ved": "low-energy / lack-of-fusion risk",
    "delta_plus_30_ved": "high-energy / keyhole-spatter risk",
}


def label(name: str) -> str:
    """Return a readable label for a class name."""
    return CLASS_LONG.get(name, name)


def prediction(scores: Dict[str, float]) -> str:
    """Return the class with maximum score."""
    return max(scores, key=scores.get)


def risk_index(scores: Dict[str, float]) -> float:
    """Combined risk score for non-stable classes."""
    return float(scores.get("delta_minus_30_ved", 0.0) + scores.get("delta_plus_30_ved", 0.0))


def score_margin(scores: Dict[str, float]) -> float:
    """Difference between top-1 and top-2 score."""
    vals = sorted([float(v) for v in scores.values()], reverse=True)
    if len(vals) < 2:
        return 1.0
    return vals[0] - vals[1]


def compute_uncertainty(
    fused_scores: Dict[str, float],
    ot_scores: Dict[str, float],
    second_scores: Dict[str, float],
) -> Tuple[str, str]:
    """Return uncertainty level and explanation.

    High uncertainty appears when:
    - OT and the second channel disagree, or
    - the fused score margin is small.
    """
    fused_margin = score_margin(fused_scores)
    ot_pred = prediction(ot_scores)
    second_pred = prediction(second_scores)

    if ot_pred != second_pred and fused_margin < 0.30:
        return (
            "HIGH",
            f"OT predicts {label(ot_pred)}, while the second channel predicts {label(second_pred)}.",
        )

    if fused_margin < 0.15:
        return (
            "HIGH",
            "The top two fused scores are very close, so the decision boundary is uncertain.",
        )

    if fused_margin < 0.30:
        return (
            "MEDIUM",
            "The fused prediction is not strongly separated from the second-best class.",
        )

    if ot_pred != second_pred:
        return (
            "MEDIUM",
            f"The sensors disagree, but the fused margin is still acceptable.",
        )

    return (
        "LOW",
        "The sensor predictions agree and the fused score has a clear margin.",
    )


def sensor_agreement_table(
    ot_scores: Dict[str, float],
    second_scores: Dict[str, float],
    fused_scores: Dict[str, float],
    second_modality: str,
) -> pd.DataFrame:
    """Summarize sensor agreement/disagreement."""
    return pd.DataFrame(
        [
            {
                "source": "OT",
                "prediction": label(prediction(ot_scores)),
                "stable": round(float(ot_scores.get("standard", 0.0)), 3),
                "low-energy": round(float(ot_scores.get("delta_minus_30_ved", 0.0)), 3),
                "high-energy": round(float(ot_scores.get("delta_plus_30_ved", 0.0)), 3),
            },
            {
                "source": second_modality.upper(),
                "prediction": label(prediction(second_scores)),
                "stable": round(float(second_scores.get("standard", 0.0)), 3),
                "low-energy": round(float(second_scores.get("delta_minus_30_ved", 0.0)), 3),
                "high-energy": round(float(second_scores.get("delta_plus_30_ved", 0.0)), 3),
            },
            {
                "source": "FUSED",
                "prediction": label(prediction(fused_scores)),
                "stable": round(float(fused_scores.get("standard", 0.0)), 3),
                "low-energy": round(float(fused_scores.get("delta_minus_30_ved", 0.0)), 3),
                "high-energy": round(float(fused_scores.get("delta_plus_30_ved", 0.0)), 3),
            },
        ]
    )


def ablation_modes_frame() -> pd.DataFrame:
    """Explain the model modes needed for a serious research study."""
    return pd.DataFrame(
        [
            {
                "mode": "rule-based demo",
                "input": "VED + simple process-state sliders",
                "purpose": "Transparent explanation and sanity check.",
                "status": "implemented",
            },
            {
                "mode": "process-feature model",
                "input": "Physics-informed descriptors",
                "purpose": "Baseline using process information only.",
                "status": "next",
            },
            {
                "mode": "sensor-descriptor model",
                "input": "OT / MPM / PBI summary descriptors",
                "purpose": "Tests whether in-situ signal summaries predict quality.",
                "status": "next",
            },
            {
                "mode": "image model",
                "input": "Raw image patches",
                "purpose": "CNN/vision model using spatial information.",
                "status": "training path exists",
            },
            {
                "mode": "hybrid fusion model",
                "input": "Process features + sensor descriptors + image scores",
                "purpose": "Most research-relevant model.",
                "status": "target",
            },
        ]
    )
