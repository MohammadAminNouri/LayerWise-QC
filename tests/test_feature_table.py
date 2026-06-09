import pandas as pd

from am_defect_detection.feature_table import build_feature_table


def test_feature_table_contains_spot_and_power_columns():
    df = pd.read_csv("data/demo_samples/manifest.csv")
    table = build_feature_table(df, image_root="data/demo_samples", include_sensor_descriptors=False)
    for col in ["phys_spot_size_mm", "phys_beam_area_mm2", "phys_power_density_w_mm2", "phys_hatch_to_spot_ratio"]:
        assert col in table.columns
