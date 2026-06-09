#!/usr/bin/env python
"""Generate a Markdown validation report."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from am_defect_detection.reporting import generate_validation_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-kind", default="unknown", choices=["unknown", "synthetic", "literature-derived", "real"])
    args = parser.parse_args()
    generate_validation_report(metrics_path=args.metrics, predictions_path=args.predictions, manifest_path=args.manifest, out_path=args.out, data_kind=args.data_kind)
    print(f"Wrote report: {args.out}")


if __name__ == "__main__":
    main()
