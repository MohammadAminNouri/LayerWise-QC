"""Sensor-view helpers for future Streamlit refactor."""
from __future__ import annotations


def modality_label(modality: str) -> str:
    return {"ot": "Optical tomography", "mpm": "Melt-pool monitoring", "pbi": "Powder-bed imaging"}.get(modality.lower(), modality)
