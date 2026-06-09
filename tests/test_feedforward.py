from am_defect_detection.feedforward_control import recommend_feedforward_control
from am_defect_detection.simulation import ProcessInputs


def test_feedforward_adjustment_is_conservative():
    current = ProcessInputs(laser_power_w=238, scan_speed_mm_s=1250, hatch_distance_mm=0.12, layer_thickness_mm=0.06)
    rec = recommend_feedforward_control(current, {"standard": 0.1, "delta_minus_30_ved": 0.8, "delta_plus_30_ved": 0.1})
    assert abs(rec.recommended_power_w - current.laser_power_w) / current.laser_power_w <= 0.071
    assert abs(rec.recommended_scan_speed_mm_s - current.scan_speed_mm_s) / current.scan_speed_mm_s <= 0.071
