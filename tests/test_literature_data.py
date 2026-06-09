import pandas as pd

from am_defect_detection.literature_data import (
    convert_literature_to_feature_table,
    infer_quality_labels,
    normalize_literature_units,
    validate_literature_records,
)


def test_literature_template_validates_and_converts():
    df = pd.read_csv("data/literature/raw_literature_records_template.csv")
    norm = infer_quality_labels(normalize_literature_units(df))
    report = validate_literature_records(norm)
    assert report.ok
    table = convert_literature_to_feature_table(norm)
    assert "citation" in table.columns
    assert "phys_ved_j_mm3" in table.columns
    assert "phys_power_density_w_mm2" in table.columns


def test_missing_citation_is_error():
    df = pd.read_csv("data/literature/raw_literature_records_template.csv")
    df.loc[0, "citation"] = ""
    report = validate_literature_records(df)
    assert not report.ok
    assert any(i.code == "empty_citation" for i in report.issues)


def test_weak_label_inference_from_defect_text():
    df = pd.DataFrame({
        "record_id": ["r1"],
        "source_id": ["s1"],
        "citation": ["citation"],
        "laser_power_w": [100],
        "scan_speed_mm_s": [1200],
        "hatch_distance_mm": [0.12],
        "layer_thickness_mm": [0.04],
        "defect_type": ["keyhole spatter"],
    })
    out = infer_quality_labels(normalize_literature_units(df))
    assert out.loc[0, "quality_label"] == "delta_plus_30_ved"
    assert out.loc[0, "label_source"] == "inferred_from_defect_text"
