#!/usr/bin/env python
"""CLI for dataset readiness auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from am_defect_detection.data_readiness import (
    audit_dataset_readiness,
    readiness_report_to_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit LPBF dataset readiness.")
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV.")
    parser.add_argument("--root", default=None, help="Dataset root directory for relative image paths.")
    parser.add_argument("--require-images", action="store_true", help="Fail if referenced image files are missing.")
    parser.add_argument("--out-md", default=None, help="Optional Markdown report output path.")
    parser.add_argument("--out-json", default=None, help="Optional JSON report output path.")
    args = parser.parse_args()

    report = audit_dataset_readiness(
        manifest_path=args.manifest,
        root_dir=args.root,
        require_images=args.require_images,
    )

    md = readiness_report_to_markdown(report)
    print(md)

    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\nSaved Markdown report to {out}")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Saved JSON report to {out}")

    return 1 if report.n_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
