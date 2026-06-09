#!/usr/bin/env python
"""Validate a LayerWise-QC manifest."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from am_defect_detection.data_manifest import validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", "--image-root", dest="root", default=None)
    parser.add_argument("--no-image-check", action="store_true")
    parser.add_argument("--require-groups", action="store_true")
    args = parser.parse_args()

    report = validate_manifest(
        args.manifest,
        root_dir=args.root,
        require_images=not args.no_image_check,
        require_group_columns=args.require_groups,
    )
    print(f"Rows: {report.n_rows}")
    print(f"OK: {report.ok}")
    print(f"Class counts: {report.class_counts}")
    print(f"Split counts: {report.split_counts}")
    if report.issues:
        print("\nIssues:")
        for issue in report.issues:
            rows = f" rows={issue.rows[:10]}" if issue.rows else ""
            print(f"[{issue.severity.upper()}] {issue.code}: {issue.message}{rows}")
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
