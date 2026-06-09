"""Metric-card helper text for the Streamlit app."""
from __future__ import annotations


def format_metric(value: float, suffix: str = "", digits: int = 2) -> str:
    return f"{value:.{digits}f}{suffix}"
