#!/usr/bin/env python
"""Make a small OT/MPM/PBI-like dataset for testing the pipeline.

This does not create experimental data. It produces sensor-like image patches
with the same label structure used in the project so the code can be run before
lab data is shared.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from am_defect_detection.constants import (  # noqa: E402
    CLASS_TO_IDX,
    PATCH_SIZES_HW,
    PROCESS_CONDITIONS,
    SPECIMEN_LAYOUT,
)
from am_defect_detection.simulation import generate_patch  # noqa: E402
from am_defect_detection.utils import ensure_dir, seed_everything  # noqa: E402


def determine_patch_class(exposure: str, layer: int, notch_start: int, defective_layers: int) -> str:
    if exposure in {"delta_minus_30_ved", "delta_plus_30_ved"}:
        if notch_start <= layer < notch_start + defective_layers:
            return exposure
    return "standard"


def parse_modalities(value: str) -> list[str]:
    modalities = [x.strip().lower() for x in value.split(",") if x.strip()]
    allowed = {"ot", "mpm", "pbi"}
    bad = sorted(set(modalities) - allowed)
    if bad:
        raise ValueError(f"Unsupported modalities: {bad}. Allowed: {sorted(allowed)}")
    if "ot" not in modalities:
        modalities.insert(0, "ot")
    return modalities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--layers", type=int, default=172, help="Valid layers after trimming.")
    parser.add_argument("--notch-start", type=int, default=70)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modalities", type=str, default="ot,mpm,pbi", help="Comma list: ot,mpm,pbi")
    args = parser.parse_args()

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    modalities = parse_modalities(args.modalities)
    out = ensure_dir(args.out)
    rows = []

    for modality in modalities:
        ensure_dir(out / modality)

    for layer in tqdm(range(args.layers), desc="synthetic layers"):
        for specimen in SPECIMEN_LAYOUT:
            specimen_id = specimen["specimen_id"]
            exposure = specimen["exposure"]
            defective_layers = specimen["defective_layers"]
            class_name = determine_patch_class(exposure, layer, args.notch_start, defective_layers)
            condition = PROCESS_CONDITIONS[class_name]
            sample_id = f"L{layer:03d}_S{specimen_id:02d}"
            paths = {}
            for modality in modalities:
                img = generate_patch(class_name, PATCH_SIZES_HW[modality], rng, modality)
                rel = Path(modality) / class_name / f"{sample_id}.jpg"
                ensure_dir(out / rel.parent)
                img.save(out / rel, quality=92)
                paths[f"{modality}_path"] = str(rel)

            rows.append(
                {
                    "sample_id": sample_id,
                    "layer": layer,
                    "specimen_id": specimen_id,
                    "source_exposure_row": exposure,
                    "defective_layers": defective_layers,
                    "class_idx": CLASS_TO_IDX[class_name],
                    "class_name": class_name,
                    "laser_power_w": condition.laser_power_w,
                    "scan_speed_mm_s": condition.scan_speed_mm_s,
                    "hatch_distance_mm": condition.hatch_distance_mm,
                    "layer_thickness_mm": condition.layer_thickness_mm,
                    "spot_size_um": condition.spot_size_um,
                    "ved_j_mm3": round(condition.ved_j_mm3, 2),
                    **paths,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out / "manifest.csv", index=False)
    print(f"Saved {len(df)} paired samples to {out / 'manifest.csv'}")
    print(df["class_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
