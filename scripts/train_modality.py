#!/usr/bin/env python
"""Train one modality-specific ResNet-18 classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from am_defect_detection.constants import PATCH_SIZES_HW  # noqa: E402
from am_defect_detection.data import (  # noqa: E402
    ModalityDataset,
    build_weighted_sampler,
    load_split_or_create,
    make_grouped_splits,
    save_splits,
    read_manifest,
)
from am_defect_detection.metrics import compute_metrics  # noqa: E402
from am_defect_detection.models import build_resnet18  # noqa: E402
from am_defect_detection.training import (  # noqa: E402
    evaluate,
    moving_average,
    predict,
    save_checkpoint,
    train_one_epoch,
)
from am_defect_detection.utils import device_from_arg, ensure_dir, save_json, seed_everything  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--modality", type=str, choices=["ot", "mpm", "pbi"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--root", type=Path, default=None, help="Base directory for relative image paths. Defaults to manifest parent.")
    parser.add_argument("--group-col", type=str, default=None, help="Optional group column for leakage-safe train/val/test splits.")
    args = parser.parse_args()

    seed_everything(args.seed)
    out = ensure_dir(args.out)
    root = args.root or args.manifest.parent
    device = device_from_arg(args.device)

    if args.group_col:
        grouped = make_grouped_splits(read_manifest(args.manifest), group_col=args.group_col, seed=args.seed)
        from am_defect_detection.data import SplitFrames
        splits = SplitFrames(
            train=grouped[grouped["split"] == "train"].reset_index(drop=True),
            val=grouped[grouped["split"] == "val"].reset_index(drop=True),
            test=grouped[grouped["split"] == "test"].reset_index(drop=True),
        )
        save_splits(splits, out)
    else:
        splits = load_split_or_create(args.manifest, out, seed=args.seed)
    patch_size = PATCH_SIZES_HW.get(args.modality, (224, 224))

    train_ds = ModalityDataset(splits.train, modality=args.modality, root=root, train=True, patch_size_hw=patch_size)
    val_ds = ModalityDataset(splits.val, modality=args.modality, root=root, train=False, patch_size_hw=patch_size)
    test_ds = ModalityDataset(splits.test, modality=args.modality, root=root, train=False, patch_size_hw=patch_size)

    sampler = build_weighted_sampler(splits.train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_resnet18(num_classes=3, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 2), gamma=0.5)

    history = []
    val_mcc_values = []
    best_score = float("-inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        val_mcc = val_metrics["matthews_corrcoef"]
        val_mcc_values.append(val_mcc)
        smoothed_mcc = moving_average(val_mcc_values, window=3)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mcc": val_mcc,
            "val_mcc_moving_average": smoothed_mcc,
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(record)
        print(record)
        if smoothed_mcc > best_score:
            best_score = smoothed_mcc
            save_checkpoint(out / "best.pt", model, optimizer, epoch, args.modality, val_metrics)

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    # Reload best model for test evaluation.
    checkpoint = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_true, y_pred, probs, sample_ids = predict(model, test_loader, device)
    test_metrics = compute_metrics(y_true, y_pred)
    save_json(test_metrics, out / "test_metrics.json")
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "y_true": y_true,
            "y_pred": y_pred,
            "p_standard": probs[:, 0],
            "p_delta_minus_30_ved": probs[:, 1],
            "p_delta_plus_30_ved": probs[:, 2],
        }
    ).to_csv(out / "test_predictions.csv", index=False)
    print("Saved best checkpoint and test metrics to", out)


if __name__ == "__main__":
    main()
