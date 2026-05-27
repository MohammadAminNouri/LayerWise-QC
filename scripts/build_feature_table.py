#!/usr/bin/env python
"""Build a feature table from a LayerWise-QC manifest.

Example
-------
python scripts/build_feature_table.py \
    --manifest data/demo_samples/manifest.csv \
    --image-root data/demo_samples \
    --out outputs/features/demo_features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from am_defect_detection.feature_table import build_feature_table, infer_feature_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV.")
    parser.add_argument("--image-root", required=True, help="Root directory for relative image paths.")
    parser.add_argument("--out", required=True, help="Output feature-table CSV path.")
    parser.add_argument(
        "--no-sensor-descriptors",
        action="store_true",
        help="Skip image descriptor extraction and export process/physics features only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    image_root = Path(args.image_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    features = build_feature_table(
        manifest,
        image_root=image_root,
        include_sensor_descriptors=not args.no_sensor_descriptors,
    )

    features.to_csv(out_path, index=False)

    groups = infer_feature_groups(features)
    print(f"Wrote feature table: {out_path}")
    print(f"Rows: {len(features)}")
    print(f"Columns: {len(features.columns)}")
    for name, cols in groups.items():
        print(f"{name}: {len(cols)} features")


if __name__ == "__main__":
    main()
