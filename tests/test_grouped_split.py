import pandas as pd

from am_defect_detection.data import make_grouped_splits, check_group_leakage


def test_grouped_split_has_no_leakage():
    df = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(12)],
        "build_id": ["a"]*3 + ["b"]*3 + ["c"]*3 + ["d"]*3,
        "class_name": ["standard", "delta_minus_30_ved", "delta_plus_30_ved"]*4,
        "class_idx": [0,1,2]*4,
    })
    out = make_grouped_splits(df, group_col="build_id")
    assert set(out["split"]).issubset({"train", "val", "test"})
    assert check_group_leakage(out, "build_id")["ok"]
