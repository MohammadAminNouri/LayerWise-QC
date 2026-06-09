"""Markdown reporting utilities for validation/evaluation outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .data_manifest import validate_manifest


def _read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_validation_report(
    *,
    metrics_path: str | Path,
    predictions_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    out_path: str | Path | None = None,
    data_kind: str = "unknown",
) -> str:
    """Generate a compact Markdown validation report."""
    metrics = _read_json(metrics_path)
    lines: list[str] = []
    lines.append("# LayerWise-QC validation report")
    lines.append("")
    lines.append(f"**Data kind:** `{data_kind}`")
    if data_kind.lower() in {"synthetic", "literature", "literature-derived"}:
        lines.append("")
        lines.append("> Warning: this report does not prove real defect-detection performance. Real claims require aligned in-situ data and independent ground truth.")

    if manifest_path:
        report = validate_manifest(manifest_path, require_images=False)
        lines += ["", "## Dataset", "", f"Rows: {report.n_rows}", f"Manifest OK: {report.ok}", f"Class counts: `{report.class_counts}`", f"Split counts: `{report.split_counts}`"]
        if report.issues:
            lines += ["", "### Manifest issues", ""]
            for issue in report.issues:
                lines.append(f"- **{issue.severity} / {issue.code}:** {issue.message}")

    lines += ["", "## Metrics", ""]
    if "results" in metrics:
        rows = []
        for key, record in metrics["results"].items():
            m = record.get("metrics", {})
            row = {"run": key, "feature_group": record.get("feature_group"), "model": record.get("model")}
            for metric_name in ["balanced_accuracy", "matthews_corrcoef", "macro_f1", "weighted_f1", "accuracy", "mae", "rmse", "r2"]:
                if metric_name in m:
                    row[metric_name] = m[metric_name]
            rows.append(row)
        if rows:
            lines.append(pd.DataFrame(rows).to_markdown(index=False))
    else:
        lines.append("```json")
        lines.append(json.dumps(metrics, indent=2)[:4000])
        lines.append("```")

    if predictions_path and Path(predictions_path).exists():
        preds = pd.read_csv(predictions_path)
        lines += ["", "## Example predictions", "", preds.head(20).to_markdown(index=False)]

    lines += ["", "## Limitations", "", "- VED and spot-size-derived features are proxies, not full thermal simulation.", "- Group-wise splitting should be used when build/specimen identifiers exist.", "- Synthetic or literature-derived results must not be presented as validated in-situ defect detection."]

    text = "\n".join(lines) + "\n"
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return text
