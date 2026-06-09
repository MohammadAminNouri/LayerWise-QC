
from pathlib import Path

import pandas as pd

from am_defect_detection.reference_profiles import (
    beam_area_mm2,
    derive_reference_from_manifest,
    normalize_against_reference,
    power_density_w_mm2,
    ved_j_mm3,
    DEMO_REFERENCE,
)


def test_ved_formula():
    v = ved_j_mm3(
        laser_power_w=200,
        scan_speed_mm_s=800,
        hatch_distance_mm=0.12,
        layer_thickness_mm=0.03,
    )
    assert abs(v - 69.4444444) < 1e-5


def test_beam_area_and_power_density():
    area = beam_area_mm2(80)
    assert area > 0
    pd = power_density_w_mm2(200, 80)
    assert pd > 0


def test_normalize_against_reference():
    out = normalize_against_reference(
        laser_power_w=200,
        scan_speed_mm_s=800,
        hatch_distance_mm=0.12,
        layer_thickness_mm=0.03,
        spot_size_um=80,
        reference=DEMO_REFERENCE,
    )
    assert "ved_reference_ratio" in out
    assert out["ved_reference_ratio"] > 0


def test_derive_reference_from_manifest(tmp_path: Path):
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "label": ["standard", "standard", "delta_plus_30_ved"],
            "material": ["Ti64", "Ti64", "Ti64"],
            "laser_power_w": [200, 210, 260],
            "scan_speed_mm_s": [800, 800, 800],
            "hatch_distance_mm": [0.12, 0.12, 0.12],
            "layer_thickness_mm": [0.03, 0.03, 0.03],
            "spot_size_um": [80, 80, 80],
        }
    )
    p = tmp_path / "manifest.csv"
    df.to_csv(p, index=False)

    profile, warnings = derive_reference_from_manifest(p)
    assert profile.reference_ved_j_mm3 > 0
    assert profile.reference_spot_size_um == 80
    assert profile.material == "Ti64"
