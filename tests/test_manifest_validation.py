from pathlib import Path
import pandas as pd

from am_defect_detection.data_manifest import validate_manifest


def test_demo_manifest_validates_with_spot_size():
    report = validate_manifest("data/demo_samples/manifest.csv", root_dir="data/demo_samples")
    assert report.ok
    assert report.n_rows > 0


def test_missing_spot_size_warns_not_error():
    df = pd.read_csv("data/demo_samples/manifest.csv").drop(columns=["spot_size_um"])
    report = validate_manifest(df, require_images=False)
    assert report.ok
    assert any(i.code == "missing_spot_size_um" and i.severity == "warning" for i in report.issues)


def test_invalid_process_value_fails():
    df = pd.read_csv("data/demo_samples/manifest.csv")
    df.loc[0, "scan_speed_mm_s"] = 0
    report = validate_manifest(df, require_images=False)
    assert not report.ok
    assert any(i.code == "invalid_scan_speed_mm_s" for i in report.issues)
