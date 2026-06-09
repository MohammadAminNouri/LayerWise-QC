"""Literature-derived benchmark data helpers.

This workflow is for traceable process/outcome records extracted from papers or
supplementary datasets. It is not a substitute for aligned in-situ sensor data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .constants import CLASS_NAMES, calculate_ved
from .data_manifest import ManifestValidationIssue, ManifestValidationReport
from .feature_table import build_feature_table


@dataclass
class LiteratureRecord:
    record_id: str
    source_id: str
    citation: str
    doi: str | None = None
    material: str | None = None
    alloy: str | None = None
    machine: str | None = None
    process: str = "LPBF"
    laser_power_w: float | None = None
    scan_speed_mm_s: float | None = None
    hatch_distance_mm: float | None = None
    layer_thickness_mm: float | None = None
    spot_size_um: float | None = None
    beam_diameter_um: float | None = None
    energy_density_j_mm3: float | None = None
    relative_density_pct: float | None = None
    porosity_pct: float | None = None
    surface_roughness_um: float | None = None
    tensile_strength_mpa: float | None = None
    elongation_pct: float | None = None
    defect_type: str | None = None
    quality_label: str | None = None
    notes: str | None = None


@dataclass
class LiteratureValidationReport:
    ok: bool
    n_rows: int
    issues: list[ManifestValidationIssue]
    label_counts: dict[str, int]

    def errors(self) -> list[ManifestValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]


def load_literature_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_literature_units(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize aliases and derive VED/spot size where possible."""
    out = df.copy()
    if "spot_size_um" not in out.columns:
        out["spot_size_um"] = np.nan
    if "beam_diameter_um" in out.columns:
        out["spot_size_um"] = out["spot_size_um"].fillna(out["beam_diameter_um"])
    out["spot_size_source"] = np.where(out["spot_size_um"].notna(), "reported_by_paper", "unknown")

    for col in ["laser_power_w", "scan_speed_mm_s", "hatch_distance_mm", "layer_thickness_mm", "spot_size_um", "energy_density_j_mm3", "relative_density_pct", "porosity_pct"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "energy_density_j_mm3" not in out.columns:
        out["energy_density_j_mm3"] = np.nan
    mask = out["energy_density_j_mm3"].isna()
    needed = ["laser_power_w", "scan_speed_mm_s", "hatch_distance_mm", "layer_thickness_mm"]
    if all(c in out.columns for c in needed):
        ok = mask
        for col in needed:
            ok &= out[col].notna() & (out[col] > 0)
        out.loc[ok, "energy_density_j_mm3"] = out.loc[ok].apply(
            lambda r: calculate_ved(r["laser_power_w"], r["scan_speed_mm_s"], r["hatch_distance_mm"], r["layer_thickness_mm"]), axis=1
        )
    return out


def infer_quality_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Infer weak quality labels only when explicit labels are missing."""
    out = df.copy()
    if "quality_label" not in out.columns:
        out["quality_label"] = pd.Series([pd.NA] * len(out), dtype="object")
    else:
        out["quality_label"] = out["quality_label"].astype("object")
    out["label_source"] = np.where(out["quality_label"].notna(), "reported_by_paper", "unknown")

    defect_series = out["defect_type"] if "defect_type" in out.columns else pd.Series("", index=out.index)
    notes_series = out["notes"] if "notes" in out.columns else pd.Series("", index=out.index)
    defect_text = (defect_series.fillna("").astype(str) + " " + notes_series.fillna("").astype(str)).str.lower()
    porosity = pd.to_numeric(out.get("porosity_pct", pd.Series(np.nan, index=out.index)), errors="coerce")
    density = pd.to_numeric(out.get("relative_density_pct", pd.Series(np.nan, index=out.index)), errors="coerce")

    missing = out["quality_label"].isna() | (out["quality_label"].astype(str).str.strip() == "")
    lof = defect_text.str.contains("lack.of.fusion|lack of fusion|lof|unmelted|insufficient", regex=True, na=False) | (porosity >= 1.0)
    keyhole = defect_text.str.contains("keyhole|spatter|balling|evaporation|overheat|excess", regex=True, na=False)
    good = ((density >= 99.0) | (porosity <= 0.5)) & ~lof & ~keyhole

    out.loc[missing & lof, "quality_label"] = "delta_minus_30_ved"
    out.loc[missing & lof, "label_source"] = "inferred_from_porosity"
    out.loc[missing & keyhole, "quality_label"] = "delta_plus_30_ved"
    out.loc[missing & keyhole, "label_source"] = "inferred_from_defect_text"
    out.loc[missing & good, "quality_label"] = "standard"
    out.loc[missing & good, "label_source"] = "inferred_from_porosity"
    return out


def validate_literature_records(df: pd.DataFrame) -> LiteratureValidationReport:
    issues: list[ManifestValidationIssue] = []

    def add(sev: Literal["error", "warning", "info"], code: str, msg: str, rows: list[int] | None = None) -> None:
        issues.append(ManifestValidationIssue(sev, code, msg, rows))

    def rows(mask: pd.Series) -> list[int]:
        return [int(i) for i in mask[mask].index.tolist()[:100]]

    for col in ["record_id", "source_id", "citation"]:
        if col not in df.columns:
            add("error", f"missing_{col}", f"Required literature column missing: {col}.")
        else:
            bad = df[col].isna() | (df[col].astype(str).str.strip() == "")
            if bad.any():
                add("error", f"empty_{col}", f"{col} is required for every literature row.", rows(bad))

    if "record_id" in df.columns:
        dup = df["record_id"].astype(str).duplicated(keep=False)
        if dup.any():
            add("error", "duplicate_record_id", "record_id must be unique.", rows(dup))

    process_cols = ["laser_power_w", "scan_speed_mm_s", "hatch_distance_mm", "layer_thickness_mm"]
    for col in process_cols:
        if col not in df.columns:
            add("warning", f"missing_{col}", f"Missing {col}; row may be unusable for process-feature modelling.")
        else:
            vals = pd.to_numeric(df[col], errors="coerce")
            bad = vals.notna() & (vals <= 0)
            if bad.any():
                add("error", f"invalid_{col}", f"{col} must be positive when reported.", rows(bad))

    if "spot_size_um" not in df.columns and "beam_diameter_um" not in df.columns:
        add("warning", "missing_spot_size", "No spot size/beam diameter column; preserve as unknown unless explicitly imputed.")
    elif "spot_size_um" in df.columns:
        vals = pd.to_numeric(df["spot_size_um"], errors="coerce")
        bad = vals.notna() & (vals <= 0)
        if bad.any():
            add("error", "invalid_spot_size_um", "spot_size_um must be positive when reported.", rows(bad))

    if "quality_label" in df.columns:
        labels = df["quality_label"].dropna().astype(str)
        invalid = ~labels.isin(CLASS_NAMES)
        if invalid.any():
            add("error", "invalid_quality_label", f"quality_label must be one of {CLASS_NAMES}.")
    else:
        add("warning", "missing_quality_label", "No quality_label column; weak labels may be inferred if outcomes/text are available.")

    outcome_cols = ["relative_density_pct", "porosity_pct", "defect_type", "quality_label"]
    if not any(c in df.columns for c in outcome_cols):
        add("warning", "missing_outcomes", "No outcome/defect columns were provided; rows cannot be supervised labels without manual labels.")

    label_counts = df.get("quality_label", pd.Series(dtype=str)).dropna().astype(str).value_counts().to_dict()
    return LiteratureValidationReport(not any(i.severity == "error" for i in issues), len(df), issues, {str(k): int(v) for k, v in label_counts.items()})


def convert_literature_to_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Convert literature CSV rows to a process-only feature table."""
    norm = infer_quality_labels(normalize_literature_units(df))
    manifest = pd.DataFrame()
    manifest["sample_id"] = norm.get("record_id", pd.Series(range(len(norm)))).astype(str)
    manifest["source_id"] = norm.get("source_id")
    manifest["citation"] = norm.get("citation")
    manifest["doi"] = norm.get("doi")
    manifest["material"] = norm.get("material")
    manifest["alloy"] = norm.get("alloy")
    manifest["machine"] = norm.get("machine")
    manifest["class_name"] = norm["quality_label"]
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    manifest["class_idx"] = manifest["class_name"].map(class_to_idx)
    manifest["label_source"] = norm.get("label_source")
    for col in ["laser_power_w", "scan_speed_mm_s", "hatch_distance_mm", "layer_thickness_mm", "spot_size_um"]:
        manifest[col] = norm.get(col)
    manifest["ved_j_mm3"] = norm.get("energy_density_j_mm3")
    manifest["relative_density"] = norm.get("relative_density_pct")
    manifest["porosity_fraction"] = pd.to_numeric(norm.get("porosity_pct"), errors="coerce") / 100.0 if "porosity_pct" in norm else np.nan
    return build_feature_table(manifest, image_root=".", include_sensor_descriptors=False, metadata_columns=list(manifest.columns))


def export_literature_manifest(df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    norm = infer_quality_labels(normalize_literature_units(df))
    manifest = pd.DataFrame({
        "sample_id": norm.get("record_id", pd.Series(range(len(norm)))).astype(str),
        "source_id": norm.get("source_id"),
        "citation": norm.get("citation"),
        "doi": norm.get("doi"),
        "class_name": norm.get("quality_label"),
        "label_source": norm.get("label_source"),
        "laser_power_w": norm.get("laser_power_w"),
        "scan_speed_mm_s": norm.get("scan_speed_mm_s"),
        "hatch_distance_mm": norm.get("hatch_distance_mm"),
        "layer_thickness_mm": norm.get("layer_thickness_mm"),
        "spot_size_um": norm.get("spot_size_um"),
        "ved_j_mm3": norm.get("energy_density_j_mm3"),
    })
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    manifest["class_idx"] = manifest["class_name"].map(class_to_idx)
    manifest.to_csv(out_path, index=False)
