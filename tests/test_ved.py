from am_defect_detection.constants import calculate_ved


def test_ved_values_match_design_table():
    assert round(calculate_ved(340, 1250, 0.12, 0.06), 2) == 37.78
    assert round(calculate_ved(238, 1250, 0.12, 0.06), 2) == 26.44
    assert round(calculate_ved(370, 1046.38, 0.12, 0.06), 2) == 49.11
