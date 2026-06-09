"""Dataset readiness checks for LPBF layer-wise quality-monitoring datasets.

This module is intentionally conservative. It does not claim that a dataset is
valid for defect detection only because files exist. It checks whether the data
is structured enough for a credible research workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Any

import math
import pandas as pd


Severity = Literal["error", "warning", "info"]


REQUIRED_ID_COLUMNS = ["sample_id"]
RECOMMENDED_GROUP_COLUMNS = ["build_id", "specimen_id", "layer_id"]
REQUIRED_PROCESS_COLUMNS = [
    "laser_power_w",
    "scan_speed_mm_s",
    "hatch_distance_mm",
    "layer_thickness_mm",
]
OPTIONAL_PROCESS_COLUMNS = [
    "spot_size_um",
    "preheat_temperature_c",
    "oxygen_ppm",
    "powder_reuse_count",
]
IMAGE_COLUMNS = ["ot_path", "mpm_path", "pbi_path"]
GROUND_TRUTH_COLUMNS = [
    "label",
    "class_name",
    "quality_label",
    "relative_density_pct",
    "porosity_pct",
    "surface_roughness_um",
    "defect_type",
]


@dataclass
class ReadinessIssue:
    severity: Severity
    code: str
    message: str
    recommendation: str | None = None


@dataclass
class DatasetReadinessReport:
    ok_for_demo: bool
    ok_for_training: bool
    ok_for_claims: bool
    n_rows: int
    n_errors: int
    n_warnings: int
    issues: list[ReadinessIssue]
    available_modalities: list[str]
    available_ground_truth: list[str]
    process_columns_present: list[str]
    group_columns_present: list[str]
    parameter_ranges: dict[str, dict[str, float]]
    class_counts: dict[str, int]
    split_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _issue(
    issues: list[ReadinessIssue],
    severity: Severity,
    code: str,
    message: str,
    recommendation: str | None = None,
) -> None:
    issues.append(ReadinessIssue(severity, code, message, recommendation))


def _numeric_range(df: pd.DataFrame, col: str) -> dict[str, float]:
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    if x.empty:
        return {}
    return {
        "min": float(x.min()),
        "median": float(x.median()),
        "max": float(x.max()),
    }


def _detect_label_column(df: pd.DataFrame) -> str | None:
    for col in ["label", "class_name", "quality_label", "defect_type"]:
        if col in df.columns:
            return col
    return None


def _check_group_leakage(df: pd.DataFrame, group_col: str, split_col: str = "split") -> dict[str, list[str]]:
    if group_col not in df.columns or split_col not in df.columns:
        return {}
    leaked: dict[str, list[str]] = {}
    for group, sub in df.groupby(group_col):
        splits = sorted(str(s) for s in sub[split_col].dropna().unique())
        if len(splits) > 1:
            leaked[str(group)] = splits
    return leaked


def audit_dataset_readiness(
    manifest_path: str | Path,
    root_dir: str | Path | None = None,
    require_images: bool = False,
) -> DatasetReadinessReport:
    """Audit a dataset manifest for research readiness.

    Parameters
    ----------
    manifest_path:
        CSV manifest containing sample IDs, process parameters, optional sensor
        paths, and ground-truth columns.
    root_dir:
        Root directory for relative image paths.
    require_images:
        If True, missing sensor image paths become errors. If False, they are
        warnings because process-only/literature workflows are still possible.
    """
    manifest_path = Path(manifest_path)
    root = Path(root_dir) if root_dir is not None else manifest_path.parent
    issues: list[ReadinessIssue] = []

    if not manifest_path.exists():
        _issue(
            issues,
            "error",
            "manifest_missing",
            f"Manifest file does not exist: {manifest_path}",
            "Check the path or upload a manifest CSV.",
        )
        return DatasetReadinessReport(
            ok_for_demo=False,
            ok_for_training=False,
            ok_for_claims=False,
            n_rows=0,
            n_errors=1,
            n_warnings=0,
            issues=issues,
            available_modalities=[],
            available_ground_truth=[],
            process_columns_present=[],
            group_columns_present=[],
            parameter_ranges={},
            class_counts={},
            split_counts={},
        )

    df = pd.read_csv(manifest_path)
    n_rows = len(df)

    if n_rows == 0:
        _issue(
            issues,
            "error",
            "empty_manifest",
            "Manifest has zero rows.",
            "Add at least one sample row.",
        )

    for col in REQUIRED_ID_COLUMNS:
        if col not in df.columns:
            _issue(
                issues,
                "error",
                f"missing_{col}",
                f"Required identifier column missing: {col}",
                "Add a unique sample_id for every sample/layer.",
            )

    if "sample_id" in df.columns:
        duplicated = int(df["sample_id"].duplicated().sum())
        if duplicated:
            _issue(
                issues,
                "error",
                "duplicate_sample_id",
                f"{duplicated} duplicated sample_id values found.",
                "Every row must have a unique sample_id.",
            )

    process_present = [c for c in REQUIRED_PROCESS_COLUMNS + OPTIONAL_PROCESS_COLUMNS if c in df.columns]

    for col in REQUIRED_PROCESS_COLUMNS:
        if col not in df.columns:
            _issue(
                issues,
                "error",
                f"missing_process_{col}",
                f"Missing required process parameter: {col}",
                "Add complete LPBF process parameters so VED and physics features can be calculated.",
            )
        else:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.isna().any():
                _issue(
                    issues,
                    "error",
                    f"non_numeric_{col}",
                    f"{int(vals.isna().sum())} non-numeric/missing values in {col}.",
                    "Fix missing or text values in numeric process columns.",
                )
            if (vals <= 0).any():
                _issue(
                    issues,
                    "error",
                    f"non_positive_{col}",
                    f"{int((vals <= 0).sum())} non-positive values in {col}.",
                    "Laser power, scan speed, hatch distance, and layer thickness must be positive.",
                )

    if "spot_size_um" not in df.columns:
        _issue(
            issues,
            "warning",
            "missing_spot_size_um",
            "spot_size_um is missing.",
            "Add laser spot size / beam diameter when available. VED alone misses beam concentration.",
        )
    else:
        spot = pd.to_numeric(df["spot_size_um"], errors="coerce")
        bad = spot.isna() | (spot <= 0)
        if bad.any():
            _issue(
                issues,
                "warning",
                "invalid_spot_size_um",
                f"{int(bad.sum())} invalid/missing spot_size_um values.",
                "Use reported beam diameter/spot size where possible; otherwise mark as missing/assumed.",
            )
        unrealistic = ((spot < 30) | (spot > 200)) & spot.notna()
        if unrealistic.any():
            _issue(
                issues,
                "warning",
                "spot_size_outside_typical_range",
                f"{int(unrealistic.sum())} spot_size_um values are outside 30–200 µm.",
                "Check units. Some papers report radius, diameter, or effective beam size differently.",
            )

    available_modalities = []
    for col in IMAGE_COLUMNS:
        if col in df.columns and df[col].notna().any():
            available_modalities.append(col.replace("_path", ""))
            if require_images:
                missing_files = 0
                for raw in df[col].dropna().astype(str):
                    if raw.strip() == "":
                        continue
                    p = Path(raw)
                    if not p.is_absolute():
                        p = root / p
                    if not p.exists():
                        missing_files += 1
                if missing_files:
                    _issue(
                        issues,
                        "error",
                        f"missing_files_{col}",
                        f"{missing_files} files referenced in {col} do not exist.",
                        "Fix paths or set the correct dataset root directory.",
                    )

    if not available_modalities:
        severity: Severity = "error" if require_images else "warning"
        _issue(
            issues,
            severity,
            "no_sensor_modalities",
            "No OT/MPM/PBI sensor image paths found.",
            "Process-only or literature mode can run, but image/sensor-fusion claims are not possible.",
        )

    gt_present = [c for c in GROUND_TRUTH_COLUMNS if c in df.columns and df[c].notna().any()]
    if not gt_present:
        _issue(
            issues,
            "error",
            "no_ground_truth",
            "No usable ground-truth/label column found.",
            "Add label, class_name, quality_label, porosity_pct, relative_density_pct, or defect_type.",
        )

    group_present = [c for c in RECOMMENDED_GROUP_COLUMNS if c in df.columns]
    if "build_id" not in df.columns:
        _issue(
            issues,
            "warning",
            "missing_build_id",
            "build_id is missing.",
            "Add build_id so train/test splitting can avoid build-level leakage.",
        )
    if "specimen_id" not in df.columns:
        _issue(
            issues,
            "warning",
            "missing_specimen_id",
            "specimen_id is missing.",
            "Add specimen_id so rows from the same specimen are not split across train/test.",
        )

    if "split" in df.columns:
        valid = {"train", "val", "validation", "test"}
        bad_splits = sorted(set(str(x) for x in df["split"].dropna().unique()) - valid)
        if bad_splits:
            _issue(
                issues,
                "error",
                "invalid_split_names",
                f"Invalid split names found: {bad_splits}",
                "Use train, val/validation, and test.",
            )
        for group_col in ["build_id", "specimen_id"]:
            leaks = _check_group_leakage(df, group_col)
            if leaks:
                _issue(
                    issues,
                    "error",
                    f"{group_col}_leakage",
                    f"{len(leaks)} {group_col} values appear in multiple splits.",
                    "Use group-wise splitting. Never let the same build/specimen appear in both train and test.",
                )
    else:
        _issue(
            issues,
            "warning",
            "missing_split",
            "No split column found.",
            "Add train/val/test split, preferably using build_id or specimen_id grouping.",
        )

    label_col = _detect_label_column(df)
    class_counts: dict[str, int] = {}
    if label_col:
        class_counts = {str(k): int(v) for k, v in df[label_col].value_counts(dropna=False).to_dict().items()}
        if len(class_counts) < 2:
            _issue(
                issues,
                "warning",
                "single_class_dataset",
                f"Only one class appears in {label_col}.",
                "Classification needs at least two classes. Use regression if you only have continuous outcomes.",
            )
        if n_rows > 0:
            smallest = min(class_counts.values())
            if smallest < 5:
                _issue(
                    issues,
                    "warning",
                    "very_small_class",
                    f"At least one class has fewer than 5 rows: {class_counts}",
                    "Collect more examples or use this only as a workflow smoke test.",
                )

    parameter_ranges = {c: _numeric_range(df, c) for c in process_present if c in df.columns}
    split_counts = {}
    if "split" in df.columns:
        split_counts = {str(k): int(v) for k, v in df["split"].value_counts(dropna=False).to_dict().items()}

    n_errors = sum(1 for i in issues if i.severity == "error")
    n_warnings = sum(1 for i in issues if i.severity == "warning")

    ok_for_demo = n_rows > 0 and n_errors == 0
    ok_for_training = (
        n_errors == 0
        and bool(gt_present)
        and all(c in df.columns for c in REQUIRED_PROCESS_COLUMNS)
        and n_rows >= 10
    )
    ok_for_claims = (
        ok_for_training
        and bool(available_modalities)
        and "build_id" in df.columns
        and "split" in df.columns
        and n_rows >= 50
        and not any("leakage" in i.code for i in issues)
    )

    if ok_for_demo and not ok_for_claims:
        _issue(
            issues,
            "info",
            "claims_limited",
            "Dataset may be usable for workflow testing, but not enough for strong real defect-detection claims.",
            "Use aligned sensor data, independent ground truth, and group-wise validation before making accuracy claims.",
        )

    n_errors = sum(1 for i in issues if i.severity == "error")
    n_warnings = sum(1 for i in issues if i.severity == "warning")

    return DatasetReadinessReport(
        ok_for_demo=ok_for_demo,
        ok_for_training=ok_for_training,
        ok_for_claims=ok_for_claims,
        n_rows=n_rows,
        n_errors=n_errors,
        n_warnings=n_warnings,
        issues=issues,
        available_modalities=available_modalities,
        available_ground_truth=gt_present,
        process_columns_present=process_present,
        group_columns_present=group_present,
        parameter_ranges=parameter_ranges,
        class_counts=class_counts,
        split_counts=split_counts,
    )


def readiness_report_to_markdown(report: DatasetReadinessReport) -> str:
    """Convert readiness report to a Markdown summary."""
    lines: list[str] = []
    lines.append("# Dataset Readiness Report")
    lines.append("")
    lines.append(f"- Rows: **{report.n_rows}**")
    lines.append(f"- Errors: **{report.n_errors}**")
    lines.append(f"- Warnings: **{report.n_warnings}**")
    lines.append(f"- OK for demo/workflow testing: **{report.ok_for_demo}**")
    lines.append(f"- OK for training experiments: **{report.ok_for_training}**")
    lines.append(f"- OK for strong accuracy/defect-detection claims: **{report.ok_for_claims}**")
    lines.append("")
    lines.append("## Available data")
    lines.append(f"- Sensor modalities: {', '.join(report.available_modalities) or 'none'}")
    lines.append(f"- Ground truth columns: {', '.join(report.available_ground_truth) or 'none'}")
    lines.append(f"- Process columns: {', '.join(report.process_columns_present) or 'none'}")
    lines.append(f"- Group columns: {', '.join(report.group_columns_present) or 'none'}")
    lines.append("")
    if report.class_counts:
        lines.append("## Class counts")
        for k, v in report.class_counts.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    if report.split_counts:
        lines.append("## Split counts")
        for k, v in report.split_counts.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    if report.parameter_ranges:
        lines.append("## Parameter ranges")
        for col, rng in report.parameter_ranges.items():
            if rng:
                lines.append(f"- {col}: min={rng['min']:.4g}, median={rng['median']:.4g}, max={rng['max']:.4g}")
        lines.append("")
    lines.append("## Issues")
    if not report.issues:
        lines.append("No issues found.")
    else:
        for issue in report.issues:
            lines.append(f"- **{issue.severity.upper()} [{issue.code}]**: {issue.message}")
            if issue.recommendation:
                lines.append(f"  - Recommendation: {issue.recommendation}")
    lines.append("")
    lines.append("## Scientific note")
    lines.append(
        "Passing this audit does not prove real defect-detection accuracy. Strong claims require aligned "
        "sensor data, independent ground truth, group-wise validation, and testing on unseen builds."
    )
    return "\n".join(lines)
