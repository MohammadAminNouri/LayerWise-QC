"""Manifest validation and dataset summary utilities for LayerWise-QC."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd


MINIMUM_COLUMNS = [
    "sample_id",
    "class_idx",
    "class_name",
    "ot_path",
    "mpm_path",
    "pbi_path",
    "laser_power_w",
    "scan_speed_mm_s",
    "hatch_distance_mm",
    "layer_thickness_mm",
    "ved_j_mm3",
]

RECOMMENDED_PROCESS_COLUMNS = [
    "spot_size_um",
]


RESEARCH_COLUMNS = [
    "build_id",
    "specimen_id",
    "layer",
    "region_id",
    "relative_density",
    "porosity_fraction",
    "ct_label",
    "metallography_label",
    "surface_defect_label",
]


def load_manifest(path_or_buffer) -> pd.DataFrame:
    """Load a CSV manifest from a path or Streamlit uploader object."""
    return pd.read_csv(path_or_buffer)


def validate_manifest_columns(
    manifest: pd.DataFrame,
    required_columns: Iterable[str] = MINIMUM_COLUMNS,
) -> Tuple[bool, List[str]]:
    """Check whether the manifest contains the required columns."""
    missing = [col for col in required_columns if col not in manifest.columns]
    return len(missing) == 0, missing


def manifest_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return a compact summary table for the app."""
    rows = [
        {"item": "rows", "value": len(manifest)},
        {"item": "columns", "value": len(manifest.columns)},
        {"item": "classes", "value": manifest["class_name"].nunique() if "class_name" in manifest else "missing"},
        {"item": "builds", "value": manifest["build_id"].nunique() if "build_id" in manifest else "not provided"},
        {"item": "specimens", "value": manifest["specimen_id"].nunique() if "specimen_id" in manifest else "not provided"},
        {"item": "layers", "value": manifest["layer"].nunique() if "layer" in manifest else "not provided"},
        {"item": "has relative density", "value": "yes" if "relative_density" in manifest else "no"},
        {"item": "has porosity fraction", "value": "yes" if "porosity_fraction" in manifest else "no"},
        {"item": "has CT label", "value": "yes" if "ct_label" in manifest else "no"},
        {"item": "has metallography label", "value": "yes" if "metallography_label" in manifest else "no"},
    ]
    return pd.DataFrame(rows)


def class_balance(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return class counts and percentages."""
    if "class_name" not in manifest.columns or manifest.empty:
        return pd.DataFrame(columns=["class_name", "count", "percentage"])

    counts = manifest["class_name"].value_counts(dropna=False).rename_axis("class_name").reset_index(name="count")
    counts["percentage"] = (counts["count"] / len(manifest) * 100.0).round(2)
    return counts


def missing_image_report(
    manifest: pd.DataFrame,
    root: str | Path,
    image_columns: Iterable[str] = ("ot_path", "mpm_path", "pbi_path"),
) -> pd.DataFrame:
    """Check missing image files relative to a root directory."""
    root = Path(root)
    rows = []

    for col in image_columns:
        if col not in manifest.columns:
            rows.append({"image column": col, "available": "no", "missing files": "column missing"})
            continue

        missing = 0
        checked = 0
        for rel in manifest[col].dropna().astype(str):
            checked += 1
            if not (root / rel).exists():
                missing += 1

        rows.append(
            {
                "image column": col,
                "available": "yes",
                "checked files": checked,
                "missing files": missing,
            }
        )

    return pd.DataFrame(rows)


def research_readiness_frame(manifest: pd.DataFrame) -> pd.DataFrame:
    """Score whether the manifest is ready for serious modelling."""
    checks = [
        {
            "requirement": "minimum process/image columns",
            "status": "yes" if validate_manifest_columns(manifest)[0] else "no",
            "why it matters": "Needed to run the existing pipeline.",
        },
        {
            "requirement": "build_id",
            "status": "yes" if "build_id" in manifest.columns else "no",
            "why it matters": "Needed for build-wise splitting and leakage control.",
        },
        {
            "requirement": "specimen_id",
            "status": "yes" if "specimen_id" in manifest.columns else "no",
            "why it matters": "Needed for specimen-wise validation.",
        },
        {
            "requirement": "layer",
            "status": "yes" if "layer" in manifest.columns else "no",
            "why it matters": "Needed for layer-wise monitoring and history.",
        },
        {
            "requirement": "laser spot size / beam diameter",
            "status": "yes" if "spot_size_um" in manifest.columns else "warning: default/unknown",
            "why it matters": "Needed for beam area, power density, and machine-to-machine comparison beyond VED.",
        },
        {
            "requirement": "ground truth density or porosity",
            "status": "yes" if {"relative_density", "porosity_fraction"} & set(manifest.columns) else "no",
            "why it matters": "Needed to move beyond process-condition labels.",
        },
        {
            "requirement": "CT/metallography/surface labels",
            "status": "yes" if {"ct_label", "metallography_label", "surface_defect_label"} & set(manifest.columns) else "no",
            "why it matters": "Needed for real defect validation.",
        },
    ]
    return pd.DataFrame(checks)


def spot_size_report(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return a compact report for optional spot-size / beam-diameter values."""
    if "spot_size_um" not in manifest.columns:
        return pd.DataFrame([{"item": "spot_size_um", "status": "missing", "message": "Optional column missing; default 80 µm will be assumed only in demo feature generation."}])

    values = pd.to_numeric(manifest["spot_size_um"], errors="coerce")
    missing = int(values.isna().sum())
    non_positive = int((values <= 0).sum())
    outside_typical = int(((values < 30) | (values > 200)).sum())
    return pd.DataFrame([
        {"item": "rows", "status": len(manifest), "message": "Rows checked."},
        {"item": "missing", "status": missing, "message": "Missing spot sizes will use the demo default only when feature generation needs a value."},
        {"item": "non_positive", "status": non_positive, "message": "Must be zero for a valid manifest."},
        {"item": "outside_30_200_um", "status": outside_typical, "message": "Warning range only; some machines may legitimately differ."},
    ])
