import math

import pandas as pd

from am_defect_detection.constants import (
    calculate_beam_area_mm2,
    calculate_power_density_w_mm2,
    spot_size_um_to_mm,
)
from am_defect_detection.feature_table import build_feature_table
from am_defect_detection.physics_features import compute_physics_features
from am_defect_detection.simulation import ProcessInputs, soft_process_scores
from am_defect_detection.data_manifest import spot_size_report


def test_spot_size_conversion_and_beam_area():
    assert spot_size_um_to_mm(80.0) == 0.08
    assert math.isclose(calculate_beam_area_mm2(80.0), math.pi * 0.04**2)


def test_power_density_uses_beam_area():
    expected = 340.0 / (math.pi * 0.04**2)
    assert math.isclose(calculate_power_density_w_mm2(340.0, 80.0), expected)


def test_physics_features_include_spot_size_terms():
    features = compute_physics_features(ProcessInputs(340, 1250, 0.12, 0.06, spot_size_um=80.0))
    for key in [
        "spot_size_um",
        "spot_size_mm",
        "beam_area_mm2",
        "power_density_w_mm2",
        "spot_overlap_ratio",
        "hatch_to_spot_ratio",
        "ved_x_power_density",
    ]:
        assert key in features
    assert math.isclose(features["spot_size_mm"], 0.08)
    assert features["power_density_w_mm2"] > 0


def test_feature_table_contains_spot_size_features_without_images(tmp_path):
    manifest = pd.DataFrame([
        {
            "sample_id": "s1",
            "class_name": "standard",
            "class_idx": 0,
            "laser_power_w": 340,
            "scan_speed_mm_s": 1250,
            "hatch_distance_mm": 0.12,
            "layer_thickness_mm": 0.06,
            "spot_size_um": 80,
        }
    ])
    table = build_feature_table(manifest, image_root=tmp_path, include_sensor_descriptors=False)
    assert "spot_size_um" in table.columns
    assert "phys_power_density_w_mm2" in table.columns
    assert "phys_hatch_to_spot_ratio" in table.columns


def test_missing_spot_size_warns_not_fails():
    manifest = pd.DataFrame({"sample_id": ["s1"]})
    report = spot_size_report(manifest)
    assert report.loc[0, "status"] == "missing"


def test_spot_size_gently_changes_proxy_scores():
    base = ProcessInputs(340, 1250, 0.12, 0.06, spot_size_um=80.0)
    small_spot = ProcessInputs(340, 1250, 0.12, 0.06, spot_size_um=45.0)
    base_scores = soft_process_scores(base)
    small_scores = soft_process_scores(small_spot)
    assert small_scores["delta_plus_30_ved"] > base_scores["delta_plus_30_ved"]
