#!/usr/bin/env python
"""Evaluate OT + MPM/PBI late-fusion soft voting."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from am_defect_detection.constants import PATCH_SIZES_HW  # noqa: E402
from am_defect_detection.data import ModalityDataset, read_manifest  # noqa: E402
from am_defect_detection.metrics import compute_metrics  # noqa: E402
from am_defect_detection.models import load_checkpoint_model  # noqa: E402
from am_defect_detection.training import predict  # noqa: E402
from am_defect_detection.utils import device_from_arg, ensure_dir, save_json  # noqa: E402


def predict_modality(checkpoint_path: Path, df: pd.DataFrame, modality: str, root: Path, device: torch.device, batch_size: int, num_workers: int):
    model, _ = load_checkpoint_model(str(checkpoint_path), device)
    ds = ModalityDataset(df, modality=modality, root=root, train=False, patch_size_hw=PATCH_SIZES_HW.get(modality, (224, 224)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    y_true, y_pred, probs, sample_ids = predict(model, loader, device)
    return y_true, y_pred, probs, sample_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ot-checkpoint", type=Path, required=True)
    parser.add_argument("--second-checkpoint", type=Path, default=None, help="MPM/PBI checkpoint. Alias: --mpm-checkpoint.")
    parser.add_argument("--mpm-checkpoint", type=Path, default=None)
    parser.add_argument("--second-modality", type=str, default="mpm", choices=["mpm", "pbi"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--w-ot", type=float, default=0.5)
    parser.add_argument("--w-second", type=float, default=0.5)
    args = parser.parse_args()

    out = ensure_dir(args.out)
    root = args.root or args.manifest.parent
    device = device_from_arg(args.device)
    second_checkpoint = args.second_checkpoint or args.mpm_checkpoint
    if second_checkpoint is None:
        raise ValueError("Provide --second-checkpoint or --mpm-checkpoint.")

    df = read_manifest(args.manifest)
    y_true_ot, _, probs_ot, sample_ids_ot = predict_modality(args.ot_checkpoint, df, "ot", root, device, args.batch_size, args.num_workers)
    y_true_second, _, probs_second, sample_ids_second = predict_modality(second_checkpoint, df, args.second_modality, root, device, args.batch_size, args.num_workers)

    if sample_ids_ot != sample_ids_second:
        raise RuntimeError("OT and second-modality predictions are not aligned by sample_id.")
    if not np.array_equal(y_true_ot, y_true_second):
        raise RuntimeError("OT and second-modality labels do not match.")

    total_w = args.w_ot + args.w_second
    probs_fused = (args.w_ot / total_w) * probs_ot + (args.w_second / total_w) * probs_second
    y_pred = probs_fused.argmax(axis=1)
    metrics = compute_metrics(y_true_ot, y_pred)
    save_json(metrics, out / "ensemble_metrics.json")
    pd.DataFrame(
        {
            "sample_id": sample_ids_ot,
            "y_true": y_true_ot,
            "y_pred_fused": y_pred,
            "p_standard": probs_fused[:, 0],
            "p_delta_minus_30_ved": probs_fused[:, 1],
            "p_delta_plus_30_ved": probs_fused[:, 2],
        }
    ).to_csv(out / "ensemble_predictions.csv", index=False)
    print("Saved ensemble metrics to", out / "ensemble_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
