#!/usr/bin/env python
"""Build process-only literature-derived benchmark features."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from am_defect_detection.literature_data import (
    convert_literature_to_feature_table,
    export_literature_manifest,
    infer_quality_labels,
    load_literature_csv,
    normalize_literature_units,
    validate_literature_records,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest-out", required=True)
    args = parser.parse_args()

    raw = load_literature_csv(args.input)
    normalized = infer_quality_labels(normalize_literature_units(raw))
    report = validate_literature_records(normalized)
    for issue in report.issues:
        print(f"[{issue.severity.upper()}] {issue.code}: {issue.message}")
    if not report.ok:
        raise SystemExit(1)

    features = convert_literature_to_feature_table(normalized)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)
    export_literature_manifest(normalized, args.manifest_out)
    print(f"Wrote literature feature table: {out}")
    print(f"Wrote literature manifest: {args.manifest_out}")
    print("WARNING: literature-derived benchmark data is for workflow testing, not real in-situ validation.")


if __name__ == "__main__":
    main()
