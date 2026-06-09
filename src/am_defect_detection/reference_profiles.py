
"""Reference profiles for LPBF process interpretation.

The values in this module are not universal material laws. They are reference
settings used to normalize process descriptors inside the app. A user should
prefer machine/material-specific values or values derived from their own
validated dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import math
import pandas as pd


@dataclass(frozen=True)
class ReferenceProfile:
    name: str
    material: str
    description: str
    reference_ved_j_mm3: float
    reference_spot_size_um: float
    reference_power_density_w_mm2: float | None = None
    reference_hatch_to_spot_ratio: float | None = None
    source: str = "user_or_demo"
    caution: str = (
        "Reference values are normalization anchors, not universal optimum values. "
        "Use material-, machine-, powder-, and validation-specific references whenever possible."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def beam_area_mm2(spot_size_um: float) -> float:
    spot_size_mm = float(spot_size_um) / 1000.0
    radius = spot_size_mm / 2.0
    return math.pi * radius * radius


def power_density_w_mm2(laser_power_w: float, spot_size_um: float) -> float:
    return float(laser_power_w) / beam_area_mm2(float(spot_size_um))


def ved_j_mm3(
    laser_power_w: float,
    scan_speed_mm_s: float,
    hatch_distance_mm: float,
    layer_thickness_mm: float,
) -> float:
    return float(laser_power_w) / (
        float(scan_speed_mm_s) * float(hatch_distance_mm) * float(layer_thickness_mm)
    )


def hatch_to_spot_ratio(hatch_distance_mm: float, spot_size_um: float) -> float:
    spot_size_mm = float(spot_size_um) / 1000.0
    return float(hatch_distance_mm) / spot_size_mm


DEMO_REFERENCE = ReferenceProfile(
    name="Demo reference used by current app",
    material="generic LPBF demo",
    description=(
        "Derived from the current demo parameter set. Use only as a transparent "
        "normalization reference for the synthetic/proxy workflow."
    ),
    reference_ved_j_mm3=37.78,
    reference_spot_size_um=80.0,
    source="current_demo_parameters",
)


REFERENCE_PROFILES: dict[str, ReferenceProfile] = {
    "demo": DEMO_REFERENCE,
    "user_defined": ReferenceProfile(
        name="User-defined reference",
        material="user-defined",
        description="User supplies the reference VED and spot size.",
        reference_ved_j_mm3=37.78,
        reference_spot_size_um=80.0,
        source="user_defined",
    ),
    "dataset_derived": ReferenceProfile(
        name="Dataset-derived reference",
        material="dataset-specific",
        description=(
            "Recommended when a manifest contains acceptable/standard samples. "
            "The app estimates the reference from those rows."
        ),
        reference_ved_j_mm3=37.78,
        reference_spot_size_um=80.0,
        source="dataset_median_of_acceptable_rows",
    ),
}


def get_reference_profile(key: str = "demo") -> ReferenceProfile:
    return REFERENCE_PROFILES.get(key, DEMO_REFERENCE)


def make_user_reference_profile(
    reference_ved_j_mm3: float,
    reference_spot_size_um: float,
    material: str = "user-defined",
    name: str = "User-defined reference",
) -> ReferenceProfile:
    return ReferenceProfile(
        name=name,
        material=material,
        description="Reference values entered by the user.",
        reference_ved_j_mm3=float(reference_ved_j_mm3),
        reference_spot_size_um=float(reference_spot_size_um),
        reference_power_density_w_mm2=None,
        reference_hatch_to_spot_ratio=None,
        source="user_defined",
    )


def classify_ved_ratio(ratio: float, low_limit: float = 0.82, high_limit: float = 1.18) -> str:
    """Classify a VED/reference ratio using conservative demo thresholds."""
    if ratio < low_limit:
        return "below_reference"
    if ratio > high_limit:
        return "above_reference"
    return "near_reference"


def normalize_against_reference(
    laser_power_w: float,
    scan_speed_mm_s: float,
    hatch_distance_mm: float,
    layer_thickness_mm: float,
    spot_size_um: float,
    reference: ReferenceProfile,
) -> dict[str, float | str]:
    current_ved = ved_j_mm3(
        laser_power_w=laser_power_w,
        scan_speed_mm_s=scan_speed_mm_s,
        hatch_distance_mm=hatch_distance_mm,
        layer_thickness_mm=layer_thickness_mm,
    )
    current_pd = power_density_w_mm2(laser_power_w, spot_size_um)
    current_hs = hatch_to_spot_ratio(hatch_distance_mm, spot_size_um)

    ref_pd = reference.reference_power_density_w_mm2
    if ref_pd is None:
        # approximate reference power-density normalization only by spot size;
        # exact value needs reference power.
        ref_pd = current_pd

    ratio = current_ved / reference.reference_ved_j_mm3

    return {
        "ved_j_mm3": current_ved,
        "reference_ved_j_mm3": reference.reference_ved_j_mm3,
        "ved_reference_ratio": ratio,
        "ved_reference_class": classify_ved_ratio(ratio),
        "spot_size_um": float(spot_size_um),
        "reference_spot_size_um": reference.reference_spot_size_um,
        "spot_size_reference_ratio": float(spot_size_um) / reference.reference_spot_size_um,
        "power_density_w_mm2": current_pd,
        "power_density_reference_ratio": current_pd / ref_pd if ref_pd else 1.0,
        "hatch_to_spot_ratio": current_hs,
    }


def _find_label_column(df: pd.DataFrame) -> str | None:
    for c in ["label", "class_name", "quality_label", "defect_type"]:
        if c in df.columns:
            return c
    return None


def _acceptable_mask(df: pd.DataFrame) -> pd.Series:
    label_col = _find_label_column(df)
    if label_col is None:
        return pd.Series([False] * len(df), index=df.index)

    s = df[label_col].astype(str).str.lower()
    keywords = [
        "standard",
        "dense",
        "acceptable",
        "nominal",
        "good",
        "ok",
        "reference",
        "low_porosity",
    ]
    return s.apply(lambda x: any(k in x for k in keywords))


def derive_reference_from_manifest(
    manifest_path: str | Path,
    prefer_acceptable_rows: bool = True,
) -> tuple[ReferenceProfile, list[str]]:
    """Derive a reference profile from a manifest.

    If acceptable/standard rows exist, the median VED of those rows is used.
    Otherwise, the median of all valid rows is used and a warning is returned.
    """
    df = pd.read_csv(manifest_path)
    warnings: list[str] = []

    required = [
        "laser_power_w",
        "scan_speed_mm_s",
        "hatch_distance_mm",
        "layer_thickness_mm",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot derive reference. Missing columns: {missing}")

    work = df.copy()
    for c in required:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    valid = (
        work["laser_power_w"].gt(0)
        & work["scan_speed_mm_s"].gt(0)
        & work["hatch_distance_mm"].gt(0)
        & work["layer_thickness_mm"].gt(0)
    )
    work = work.loc[valid].copy()

    if work.empty:
        raise ValueError("Cannot derive reference. No valid positive process rows found.")

    work["ved_j_mm3"] = (
        work["laser_power_w"]
        / (work["scan_speed_mm_s"] * work["hatch_distance_mm"] * work["layer_thickness_mm"])
    )

    selected = work
    if prefer_acceptable_rows:
        mask = _acceptable_mask(work)
        if mask.any():
            selected = work.loc[mask].copy()
        else:
            warnings.append(
                "No acceptable/standard label found. Reference was derived from all valid rows, not from confirmed good parts."
            )

    ref_ved = float(selected["ved_j_mm3"].median())

    if "spot_size_um" in selected.columns:
        spot = pd.to_numeric(selected["spot_size_um"], errors="coerce")
        spot = spot[spot.gt(0)]
        if len(spot):
            ref_spot = float(spot.median())
        else:
            ref_spot = 80.0
            warnings.append("spot_size_um exists but has no valid positive values. Used 80 µm as demo fallback.")
    else:
        ref_spot = 80.0
        warnings.append("spot_size_um is missing. Used 80 µm as demo fallback.")

    profile = ReferenceProfile(
        name="Dataset-derived reference",
        material=str(df["material"].dropna().iloc[0]) if "material" in df.columns and df["material"].notna().any() else "dataset-specific",
        description="Median reference derived from manifest rows.",
        reference_ved_j_mm3=ref_ved,
        reference_spot_size_um=ref_spot,
        source="manifest_median",
        caution=(
            "This is a dataset-specific reference. It is stronger when derived from validated acceptable parts. "
            "It should not be transferred to another material, machine, powder batch, or optical setup without validation."
        ),
    )
    return profile, warnings
