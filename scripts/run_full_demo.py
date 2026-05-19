#!/usr/bin/env python
"""Run a small end-to-end demo."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    data_dir = args.out / "synthetic"
    manifest = data_dir / "manifest.csv"
    py = sys.executable

    run([py, "scripts/make_synthetic_dataset.py", "--out", str(data_dir), "--layers", str(args.layers), "--notch-start", str(max(5, args.layers // 2))])

    base_train = ["--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--num-workers", "0"]
    if args.device:
        base_train += ["--device", args.device]
    if args.no_pretrained:
        base_train += ["--no-pretrained"]

    run([py, "scripts/train_modality.py", "--manifest", str(manifest), "--modality", "ot", "--out", str(args.out / "ot_run"), *base_train])
    run([py, "scripts/train_modality.py", "--manifest", str(manifest), "--modality", "mpm", "--out", str(args.out / "mpm_run"), *base_train])
    eval_cmd = [
        py,
        "scripts/evaluate_ensemble.py",
        "--manifest",
        str(manifest),
        "--ot-checkpoint",
        str(args.out / "ot_run" / "best.pt"),
        "--mpm-checkpoint",
        str(args.out / "mpm_run" / "best.pt"),
        "--out",
        str(args.out / "ensemble_eval"),
        "--num-workers",
        "0",
    ]
    if args.device:
        eval_cmd += ["--device", args.device]
    run(eval_cmd)
    run([py, "scripts/run_gradcam.py", "--manifest", str(manifest), "--checkpoint", str(args.out / "ot_run" / "best.pt"), "--modality", "ot", "--out", str(args.out / "gradcam"), "--row", "0"])


if __name__ == "__main__":
    main()
