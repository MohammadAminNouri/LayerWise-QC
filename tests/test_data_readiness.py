
from pathlib import Path

import pandas as pd

from am_defect_detection.data_readiness import (
    audit_dataset_readiness,
    readiness_report_to_markdown,
)


def test_readiness_valid_minimal_manifest(tmp_path: Path):
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            "build_id": ["b1"] * 5 + ["b2"] * 5,
            "specimen_id": [f"s{i}" for i in range(10)],
            "split": ["train"] * 5 + ["test"] * 5,
            "laser_power_w": [200] * 10,
            "scan_speed_mm_s": [800] * 10,
            "hatch_distance_mm": [0.1] * 10,
            "layer_thickness_mm": [0.03] * 10,
            "spot_size_um": [80] * 10,
            "label": ["standard"] * 5 + ["delta_minus_30_ved"] * 5,
        }
    )
    p = tmp_path / "manifest.csv"
    df.to_csv(p, index=False)

    report = audit_dataset_readiness(p)

    assert report.n_errors == 0
    assert report.ok_for_training
    assert "spot_size_um" in report.process_columns_present


def test_readiness_catches_missing_ground_truth(tmp_path: Path):
    df = pd.DataFrame(
        {
            "sample_id": ["a"],
            "laser_power_w": [200],
            "scan_speed_mm_s": [800],
            "hatch_distance_mm": [0.1],
            "layer_thickness_mm": [0.03],
        }
    )
    p = tmp_path / "manifest.csv"
    df.to_csv(p, index=False)

    report = audit_dataset_readiness(p)

    assert report.n_errors > 0
    assert any(i.code == "no_ground_truth" for i in report.issues)


def test_readiness_catches_group_leakage(tmp_path: Path):
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "build_id": ["build_1", "build_1"],
            "specimen_id": ["s1", "s2"],
            "split": ["train", "test"],
            "laser_power_w": [200, 210],
            "scan_speed_mm_s": [800, 820],
            "hatch_distance_mm": [0.1, 0.1],
            "layer_thickness_mm": [0.03, 0.03],
            "label": ["standard", "standard"],
        }
    )
    p = tmp_path / "manifest.csv"
    df.to_csv(p, index=False)

    report = audit_dataset_readiness(p)

    assert any("leakage" in i.code for i in report.issues)


def test_readiness_markdown_contains_ground_truth_note(tmp_path: Path):
    df = pd.DataFrame(
        {
            "sample_id": ["a"],
            "laser_power_w": [200],
            "scan_speed_mm_s": [800],
            "hatch_distance_mm": [0.1],
            "layer_thickness_mm": [0.03],
            "label": ["standard"],
        }
    )
    p = tmp_path / "manifest.csv"
    df.to_csv(p, index=False)

    report = audit_dataset_readiness(p)
    md = readiness_report_to_markdown(report)

    assert "Dataset Readiness Report" in md
    assert "ground truth" in md.lower()
