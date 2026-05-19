#!/usr/bin/env python
"""Generate a Grad-CAM overlay for one manifest sample."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from am_defect_detection.constants import PATCH_SIZES_HW, IDX_TO_CLASS  # noqa: E402
from am_defect_detection.data import ModalityDataset, read_manifest  # noqa: E402
from am_defect_detection.gradcam import GradCAM, save_gradcam_overlay  # noqa: E402
from am_defect_detection.models import load_checkpoint_model  # noqa: E402
from am_defect_detection.utils import device_from_arg, ensure_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--modality", type=str, choices=["ot", "mpm", "pbi"], required=True)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = device_from_arg(args.device)
    root = args.root or args.manifest.parent
    df = read_manifest(args.manifest)
    if args.sample_id is not None:
        df = df[df["sample_id"].astype(str) == args.sample_id]
        if df.empty:
            raise ValueError(f"sample_id not found: {args.sample_id}")
    else:
        df = df.iloc[[args.row]]

    model, _ = load_checkpoint_model(str(args.checkpoint), device)
    ds = ModalityDataset(df, modality=args.modality, root=root, train=False, patch_size_hw=PATCH_SIZES_HW.get(args.modality, (224, 224)))
    item = ds[0]
    image_tensor = item["image"].unsqueeze(0).to(device)

    target_layer = model.layer4[-1]
    cam_runner = GradCAM(model, target_layer)
    cam, class_idx = cam_runner(image_tensor)
    out_path = ensure_dir(args.out) / f"gradcam_{args.modality}_{item['sample_id']}_{IDX_TO_CLASS[class_idx]}.png"
    save_gradcam_overlay(image_tensor, cam, out_path)
    print(f"Saved Grad-CAM overlay: {out_path}")


if __name__ == "__main__":
    main()
