"""Manifest validation and dataset summary utilities for LayerWise-QC.

This module intentionally keeps the old lightweight helpers while adding a
structured validator suitable for real/literature/synthetic dataset checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, List, Tuple

import pandas as pd

from .constants import CLASS_NAMES

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

RECOMMENDED_PROCESS_COLUMNS = ["spot_size_um"]

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

PROCESS_COLUMNS = [
    "laser_power_w",
    "scan_speed_mm_s",
    "hatch_distance_mm",
    "layer_thickness_mm",
]

IMAGE_COLUMNS = ("ot_path", "mpm_path", "pbi_path")
GROUP_COLUMNS = ("build_id", "specimen_id")


@dataclass
class ManifestValidationIssue:
    severity: Literal["error", "warning", "info"]
    code: str
    message: str
    rows: list[int] | None = None


@dataclass
class ManifestValidationReport:
    ok: bool
    n_rows: int
    issues: list[ManifestValidationIssue]
    class_counts: dict[str, int]
    split_counts: dict[str, int]

    def errors(self) -> list[ManifestValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    def warnings(self) -> list[ManifestValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([i.__dict__ for i in self.issues])


def load_manifest(path_or_buffer) -> pd.DataFrame:
    """Load a CSV manifest from a path or Streamlit uploader object."""
    return pd.read_csv(path_or_buffer)


def _label_column(df: pd.DataFrame) -> str | None:
    for col in ("class_name", "label", "quality_label"):
        if col in df.columns:
            return col
    return None


def _rows_where(mask: pd.Series) -> list[int]:
    return [int(i) for i in mask[mask].index.tolist()[:100]]


def validate_manifest(
    manifest_path: str | Path | pd.DataFrame,
    root_dir: str | Path | None = None,
    require_images: bool = True,
    require_process_params: bool = True,
    require_group_columns: bool = False,
) -> ManifestValidationReport:
    """Validate a LayerWise-QC manifest.

    Missing ``spot_size_um`` is a warning, not an error, to preserve older demo
    manifests. Existing manifests can still run with the demo default, but real
    reports should mark that assumption.
    """
    df = manifest_path.copy() if isinstance(manifest_path, pd.DataFrame) else pd.read_csv(manifest_path)
    issues: list[ManifestValidationIssue] = []

    def add(severity: Literal["error", "warning", "info"], code: str, message: str, rows: list[int] | None = None) -> None:
        issues.append(ManifestValidationIssue(severity, code, message, rows))

    if df.empty:
        add("error", "empty_manifest", "Manifest has zero rows.")

    if "sample_id" not in df.columns:
        add("error", "missing_sample_id", "Required column missing: sample_id.")
    else:
        dup = df["sample_id"].astype(str).duplicated(keep=False)
        if dup.any():
            add("error", "duplicate_sample_id", "sample_id must be unique.", _rows_where(dup))

    label_col = _label_column(df)
    if label_col is None:
        add("error", "missing_label", "Required label column missing: class_name, label, or quality_label.")
        class_counts: dict[str, int] = {}
    else:
        labels = df[label_col].astype(str)
        class_counts = labels.value_counts(dropna=False).to_dict()
        invalid = ~labels.isin(CLASS_NAMES)
        if invalid.any():
            add("error", "invalid_label", f"Labels must be one of {CLASS_NAMES}.", _rows_where(invalid))
        if len(class_counts) > 1 and min(class_counts.values()) < 3:
            add("warning", "class_imbalance", f"Some classes have fewer than 3 rows: {class_counts}.")

    if require_process_params:
        for col in PROCESS_COLUMNS:
            if col not in df.columns:
                add("error", f"missing_{col}", f"Required process column missing: {col}.")
            else:
                values = pd.to_numeric(df[col], errors="coerce")
                bad = values.isna() | (values <= 0)
                if bad.any():
                    add("error", f"invalid_{col}", f"{col} must be positive numeric.", _rows_where(bad))

    if "spot_size_um" not in df.columns:
        add("warning", "missing_spot_size_um", "spot_size_um missing; demo feature generation may assume 80 µm. Real datasets should report or explicitly mark imputation.")
    else:
        values = pd.to_numeric(df["spot_size_um"], errors="coerce")
        bad = values.notna() & (values <= 0)
        if bad.any():
            add("error", "invalid_spot_size_um", "spot_size_um must be positive when provided.", _rows_where(bad))
        unusual = values.notna() & ((values < 30) | (values > 200))
        if unusual.any():
            add("warning", "unusual_spot_size_um", "spot_size_um outside typical 30–200 µm range; verify machine metadata.", _rows_where(unusual))

    if "split" in df.columns:
        split = df["split"].astype(str).str.lower()
        invalid_split = ~split.isin(["train", "val", "test"])
        if invalid_split.any():
            add("error", "invalid_split", "split must be train, val, or test.", _rows_where(invalid_split))
        split_counts = split.value_counts(dropna=False).to_dict()
        for group_col in GROUP_COLUMNS:
            if group_col in df.columns:
                leaked = []
                for group, g in df.groupby(group_col, dropna=True):
                    splits = set(g["split"].dropna().astype(str).str.lower())
                    if len(splits) > 1:
                        leaked.append(str(group))
                if leaked:
                    add("warning", f"{group_col}_leakage_risk", f"{group_col} appears in multiple splits: {leaked[:10]}.")
    else:
        split_counts = {}
        add("info", "missing_split", "No split column; training scripts will create splits.")

    if require_group_columns:
        for col in GROUP_COLUMNS:
            if col not in df.columns:
                add("error", f"missing_{col}", f"Required group column missing for leakage-safe validation: {col}.")
    else:
        missing_groups = [c for c in GROUP_COLUMNS if c not in df.columns]
        if missing_groups:
            add("warning", "missing_group_columns", f"Missing group columns {missing_groups}; leakage-safe splitting cannot be guaranteed.")

    if require_images:
        root = Path(root_dir) if root_dir else None
        for col in IMAGE_COLUMNS:
            if col not in df.columns:
                add("warning", f"missing_{col}", f"Image path column missing: {col}.")
                continue
            empty = df[col].isna() | (df[col].astype(str).str.strip() == "")
            if empty.any():
                add("warning", f"empty_{col}", f"Some rows have empty {col}.", _rows_where(empty))
            if root is not None:
                missing = []
                for idx, value in df[col].dropna().items():
                    p = Path(str(value))
                    candidate = p if p.is_absolute() else root / p
                    if not candidate.exists():
                        missing.append(int(idx))
                if missing:
                    add("warning", f"missing_files_{col}", f"{len(missing)} files referenced by {col} do not exist under {root}.", missing[:100])

    return ManifestValidationReport(
        ok=not any(i.severity == "error" for i in issues),
        n_rows=int(len(df)),
        issues=issues,
        class_counts={str(k): int(v) for k, v in class_counts.items()},
        split_counts={str(k): int(v) for k, v in split_counts.items()},
    )


def validate_manifest_columns(manifest: pd.DataFrame, required_columns: Iterable[str] = MINIMUM_COLUMNS) -> Tuple[bool, List[str]]:
    missing = [col for col in required_columns if col not in manifest.columns]
    return len(missing) == 0, missing


def manifest_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"item": "rows", "value": len(manifest)},
        {"item": "columns", "value": len(manifest.columns)},
        {"item": "classes", "value": manifest["class_name"].nunique() if "class_name" in manifest else "missing"},
        {"item": "builds", "value": manifest["build_id"].nunique() if "build_id" in manifest else "not provided"},
        {"item": "specimens", "value": manifest["specimen_id"].nunique() if "specimen_id" in manifest else "not provided"},
        {"item": "layers", "value": manifest["layer"].nunique() if "layer" in manifest else "not provided"},
        {"item": "has spot size", "value": "yes" if "spot_size_um" in manifest else "warning: assumed/unknown"},
        {"item": "has relative density", "value": "yes" if "relative_density" in manifest else "no"},
        {"item": "has porosity fraction", "value": "yes" if "porosity_fraction" in manifest else "no"},
        {"item": "has CT label", "value": "yes" if "ct_label" in manifest else "no"},
        {"item": "has metallography label", "value": "yes" if "metallography_label" in manifest else "no"},
    ]
    return pd.DataFrame(rows)


def class_balance(manifest: pd.DataFrame) -> pd.DataFrame:
    if "class_name" not in manifest.columns or manifest.empty:
        return pd.DataFrame(columns=["class_name", "count", "percentage"])
    counts = manifest["class_name"].value_counts(dropna=False).rename_axis("class_name").reset_index(name="count")
    counts["percentage"] = (counts["count"] / len(manifest) * 100.0).round(2)
    return counts


def missing_image_report(manifest: pd.DataFrame, root: str | Path, image_columns: Iterable[str] = IMAGE_COLUMNS) -> pd.DataFrame:
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
        rows.append({"image column": col, "available": "yes", "checked files": checked, "missing files": missing})
    return pd.DataFrame(rows)


def research_readiness_frame(manifest: pd.DataFrame) -> pd.DataFrame:
    checks = [
        {"requirement": "minimum process/image columns", "status": "yes" if validate_manifest_columns(manifest)[0] else "no", "why it matters": "Needed to run the existing pipeline."},
        {"requirement": "build_id", "status": "yes" if "build_id" in manifest.columns else "no", "why it matters": "Needed for build-wise splitting and leakage control."},
        {"requirement": "specimen_id", "status": "yes" if "specimen_id" in manifest.columns else "no", "why it matters": "Needed for specimen-wise validation."},
        {"requirement": "layer", "status": "yes" if "layer" in manifest.columns else "no", "why it matters": "Needed for layer-wise monitoring and history."},
        {"requirement": "laser spot size / beam diameter", "status": "yes" if "spot_size_um" in manifest.columns else "warning: default/unknown", "why it matters": "Needed for beam area, power density, and machine-to-machine comparison beyond VED."},
        {"requirement": "ground truth density or porosity", "status": "yes" if {"relative_density", "porosity_fraction"} & set(manifest.columns) else "no", "why it matters": "Needed to move beyond process-condition labels."},
        {"requirement": "CT/metallography/surface labels", "status": "yes" if {"ct_label", "metallography_label", "surface_defect_label"} & set(manifest.columns) else "no", "why it matters": "Needed for real defect validation."},
    ]
    return pd.DataFrame(checks)


def spot_size_report(manifest: pd.DataFrame) -> pd.DataFrame:
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
