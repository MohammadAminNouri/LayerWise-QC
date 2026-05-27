"""Sensor-image descriptors for LayerWise-QC.

The app currently uses synthetic demo images. These functions make the synthetic
images more useful by extracting interpretable descriptors that can also be
applied later to real OT, MPM, or PBI images.

Examples:
- mode intensity,
- interquartile range,
- variance,
- hot/cold pixel fraction,
- entropy,
- texture energy,
- streakiness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from PIL import Image


def image_to_gray_array(image_or_path: Any) -> np.ndarray:
    """Convert a PIL image, numpy array, or path into a grayscale float array."""
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


def empty_sensor_descriptors() -> Dict[str, float]:
    """Return zero descriptors for an empty or invalid image."""
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


def compute_sensor_descriptors(image_or_path: Any) -> Dict[str, float]:
    """Extract simple image descriptors from an OT, MPM, or PBI image."""
    arr = image_to_gray_array(image_or_path)
    flat = arr.ravel()

    if flat.size == 0:
        return empty_sensor_descriptors()

    hist, bin_edges = np.histogram(flat, bins=64, range=(0, 255))
    mode_bin = int(np.argmax(hist))
    mode_intensity = float((bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2)

    p05, p25, p50, p75, p95, p99 = np.percentile(
        flat,
        [5, 25, 50, 75, 95, 99],
    )

    prob = hist.astype(np.float64)
    prob = prob / max(prob.sum(), 1.0)
    histogram_entropy = -float(np.sum(prob[prob > 0] * np.log2(prob[prob > 0])))

    if arr.ndim != 2:
        arr = arr.reshape(-1, 1)

    grad_y_mean = float(np.mean(np.abs(np.diff(arr, axis=0)))) if arr.shape[0] > 1 else 0.0
    grad_x_mean = float(np.mean(np.abs(np.diff(arr, axis=1)))) if arr.shape[1] > 1 else 0.0
    texture_energy = grad_x_mean + grad_y_mean

    # Useful for powder-bed imaging: strong row-to-row variation can mimic streaks.
    streakiness = float(np.std(arr.mean(axis=1))) if arr.shape[0] > 1 else 0.0

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
        "histogram_entropy": float(histogram_entropy),
        "texture_energy": float(texture_energy),
        "streakiness": float(streakiness),
    }


def sensor_descriptor_explanations() -> Dict[str, str]:
    """Human-readable meaning of sensor descriptors."""
    return {
        "mean_intensity": "Average image intensity. For thermal signals, higher values may indicate stronger emission.",
        "median_intensity": "Middle image intensity value. Less sensitive to extreme hot pixels than the mean.",
        "mode_intensity": "Most frequent intensity range. Useful for dominant thermal/image background.",
        "variance_intensity": "Spread of intensity values. High variance can indicate non-uniformity.",
        "std_intensity": "Standard deviation of intensity values.",
        "iqr_intensity": "Interquartile range. Robust measure of signal spread.",
        "p05_intensity": "Low-end intensity percentile.",
        "p95_intensity": "High-end intensity percentile.",
        "p99_intensity": "Extreme high-end intensity percentile.",
        "hot_pixel_fraction": "Fraction of pixels in the top 5% intensity range.",
        "cold_pixel_fraction": "Fraction of pixels in the bottom 5% intensity range.",
        "histogram_entropy": "Intensity-distribution complexity.",
        "texture_energy": "Simple gradient-based texture/non-uniformity measure.",
        "streakiness": "Row-wise variation, useful for powder-bed streak or recoater-mark proxies.",
    }


def sensor_descriptors_to_frame(
    descriptors_by_modality: Dict[str, Dict[str, float]],
    *,
    include_explanations: bool = True,
) -> pd.DataFrame:
    """Convert nested descriptors into a Streamlit-friendly table."""
    explanations = sensor_descriptor_explanations()
    rows = []
    for modality, desc in descriptors_by_modality.items():
        for key, value in desc.items():
            row = {
                "modality": modality.upper(),
                "descriptor": key,
                "value": round(float(value), 5),
            }
            if include_explanations:
                row["why it matters"] = explanations.get(key, "")
            rows.append(row)
    return pd.DataFrame(rows)
