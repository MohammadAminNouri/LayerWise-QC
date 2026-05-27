"""Build feature tables for LayerWise-QC training.

This module converts a manifest into a machine-learning table.

Input:
    A CSV manifest with process columns and optional sensor image paths.

Output:
    One row per manifest row, containing:
    - metadata columns,
    - process parameters,
    - physics-informed descriptors,
    - OT / MPM / PBI image descriptors,
    - target labels if available.

This is the bridge between the Streamlit prototype and real training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .physics_features import compute_physics_features
from .sensor_features import compute_sensor_descriptors
from .simulation import ProcessInputs


PROCESS_COLUMNS = [
    "laser_power_w",
    "scan_speed_mm_s",
    "hatch_distance_mm",
    "layer_thickness_mm",
]

OPTIONAL_STATE_COLUMNS = [
    "heat_memory",
    "powder_uniformity",
]

IMAGE_PATH_COLUMNS = {
    "ot": "ot_path",
    "mpm": "mpm_path",
    "pbi": "pbi_path",
}

DEFAULT_METADATA_COLUMNS = [
    "sample_id",
    "build_id",
    "specimen_id",
    "layer",
    "region_id",
    "class_idx",
    "class_name",
    "relative_density",
    "porosity_fraction",
    "ct_label",
    "metallography_label",
    "surface_defect_label",
]


def _safe_float(row: pd.Series, key: str, default: float) -> float:
    value = row.get(key, default)
    if pd.isna(value):
        return float(default)
    return float(value)


def process_inputs_from_row(row: pd.Series) -> ProcessInputs:
    """Create ProcessInputs from a manifest row."""
    return ProcessInputs(
        laser_power_w=_safe_float(row, "laser_power_w", 340.0),
        scan_speed_mm_s=_safe_float(row, "scan_speed_mm_s", 1250.0),
        hatch_distance_mm=_safe_float(row, "hatch_distance_mm", 0.12),
        layer_thickness_mm=_safe_float(row, "layer_thickness_mm", 0.06),
        heat_memory=_safe_float(row, "heat_memory", 0.35),
        powder_uniformity=_safe_float(row, "powder_uniformity", 0.86),
    )


def _existing_image_path(root: Path, rel_path: object) -> Path | None:
    if rel_path is None or pd.isna(rel_path):
        return None

    path = Path(str(rel_path))
    if path.is_absolute() and path.exists():
        return path

    candidate = root / path
    if candidate.exists():
        return candidate

    return None


def build_feature_table(
    manifest: pd.DataFrame,
    *,
    image_root: str | Path,
    include_sensor_descriptors: bool = True,
    metadata_columns: Iterable[str] = DEFAULT_METADATA_COLUMNS,
) -> pd.DataFrame:
    """Build a feature table from a manifest dataframe.

    Parameters
    ----------
    manifest:
        Manifest dataframe.
    image_root:
        Directory used to resolve relative image paths.
    include_sensor_descriptors:
        If true, image descriptors are computed for OT / MPM / PBI when files exist.
    metadata_columns:
        Columns copied from the manifest when present.

    Returns
    -------
    pd.DataFrame
        ML-ready feature table.
    """

    image_root = Path(image_root)
    rows: list[dict[str, object]] = []

    for _, row in manifest.iterrows():
        inputs = process_inputs_from_row(row)

        out: dict[str, object] = {}

        # Copy useful metadata and targets when available.
        for col in metadata_columns:
            if col in manifest.columns:
                out[col] = row.get(col)

        # Copy raw process parameters.
        for col in PROCESS_COLUMNS + OPTIONAL_STATE_COLUMNS:
            if col in manifest.columns:
                out[col] = row.get(col)

        # Add physics-informed features.
        physics = compute_physics_features(inputs)
        for key, value in physics.items():
            out[f"phys_{key}"] = value

        # Add sensor descriptors from image paths.
        if include_sensor_descriptors:
            for modality, path_col in IMAGE_PATH_COLUMNS.items():
                if path_col not in manifest.columns:
                    continue

                image_path = _existing_image_path(image_root, row.get(path_col))
                out[f"{modality}_image_found"] = image_path is not None

                if image_path is None:
                    continue

                try:
                    desc = compute_sensor_descriptors(image_path)
                except Exception as exc:  # defensive: keep one bad image from breaking all data
                    out[f"{modality}_descriptor_error"] = str(exc)
                    continue

                for key, value in desc.items():
                    out[f"{modality}_{key}"] = value

        rows.append(out)

    return pd.DataFrame(rows)


def infer_feature_groups(table: pd.DataFrame) -> dict[str, list[str]]:
    """Infer feature groups for ablation models."""
    numeric_cols = [
        col for col in table.columns
        if pd.api.types.is_numeric_dtype(table[col])
    ]

    forbidden_prefixes = (
        "class_idx",
    )
    forbidden_cols = {
        "relative_density",
        "porosity_fraction",
        "layer",
        "build_id",
        "specimen_id",
        "sample_id",
        "region_id",
    }

    candidate_cols = [
        col for col in numeric_cols
        if col not in forbidden_cols and not col.startswith(forbidden_prefixes)
    ]

    process = [
        col for col in candidate_cols
        if col in PROCESS_COLUMNS + OPTIONAL_STATE_COLUMNS or col.startswith("phys_")
    ]

    sensor = [
        col for col in candidate_cols
        if col.startswith(("ot_", "mpm_", "pbi_"))
        and not col.endswith("_image_found")
    ]

    hybrid = sorted(set(process + sensor))

    return {
        "process_only": process,
        "sensor_descriptors_only": sensor,
        "hybrid_process_sensor": hybrid,
    }
