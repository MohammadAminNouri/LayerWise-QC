#!/usr/bin/env python
"""Create a larger synthetic manifest for stress-testing the training pipeline.

This does not replace real data. It only helps test:
- feature extraction,
- group splits,
- class balance,
- training scripts,
- app manifest validation.

It assumes your existing repository already has image generation through:
    am_defect_detection.simulation.image_from_process
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from am_defect_detection.constants import PATCH_SIZES_HW
from am_defect_detection.simulation import ProcessInputs, classify_from_ved, image_from_process


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output dataset directory.")
    p.add_argument("--builds", type=int, default=6)
    p.add_argument("--specimens-per-build", type=int, default=12)
    p.add_argument("--layers", type=int, default=25)
    p.add_argument("--regions-per-layer", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sample_inputs(rng: np.random.Generator, condition: str) -> ProcessInputs:
    if condition == "standard":
        power = rng.normal(340, 18)
        speed = rng.normal(1250, 80)
        heat = rng.uniform(0.25, 0.55)
        powder = rng.uniform(0.75, 0.95)
    elif condition == "delta_minus_30_ved":
        power = rng.normal(250, 20)
        speed = rng.normal(1375, 100)
        heat = rng.uniform(0.15, 0.45)
        powder = rng.uniform(0.45, 0.85)
    else:
        power = rng.normal(385, 22)
        speed = rng.normal(1100, 95)
        heat = rng.uniform(0.55, 0.90)
        powder = rng.uniform(0.70, 0.95)

    hatch = float(np.clip(rng.normal(0.12, 0.008), 0.08, 0.17))
    layer = float(np.clip(rng.normal(0.06, 0.005), 0.03, 0.09))

    return ProcessInputs(
        laser_power_w=float(np.clip(power, 180, 430)),
        scan_speed_mm_s=float(np.clip(speed, 650, 1700)),
        hatch_distance_mm=hatch,
        layer_thickness_mm=layer,
        heat_memory=float(np.clip(heat, 0, 1)),
        powder_uniformity=float(np.clip(powder, 0, 1)),
    )


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    class_names = ["standard", "delta_minus_30_ved", "delta_plus_30_ved"]

    for build in range(args.builds):
        build_id = f"build_{build:03d}"

        for specimen in range(args.specimens_per_build):
            specimen_id = f"{build_id}_specimen_{specimen:03d}"
            nominal_condition = class_names[(build + specimen) % len(class_names)]

            for layer in range(args.layers):
                for region in range(args.regions_per_layer):
                    # Occasionally perturb condition to make layer/region variation.
                    condition = nominal_condition
                    if rng.random() < 0.10:
                        condition = rng.choice(class_names)

                    inputs = sample_inputs(rng, condition)
                    rule_condition = classify_from_ved(inputs.ved)

                    sample_id = f"{specimen_id}_L{layer:04d}_R{region:02d}"
                    row = {
                        "sample_id": sample_id,
                        "build_id": build_id,
                        "specimen_id": specimen_id,
                        "layer": layer,
                        "region_id": region,
                        "class_name": condition,
                        "class_idx": class_names.index(condition),
                        "rule_class_name": rule_condition,
                        "laser_power_w": inputs.laser_power_w,
                        "scan_speed_mm_s": inputs.scan_speed_mm_s,
                        "hatch_distance_mm": inputs.hatch_distance_mm,
                        "layer_thickness_mm": inputs.layer_thickness_mm,
                        "heat_memory": inputs.heat_memory,
                        "powder_uniformity": inputs.powder_uniformity,
                        "ved_j_mm3": inputs.ved,
                    }

                    # Synthetic quality labels. These are only proxies.
                    nved = inputs.ved / 37.78
                    porosity = 0.005 + 0.05 * max(0, 0.85 - nved) + 0.035 * max(0, nved - 1.15)
                    porosity += 0.02 * max(0, 0.70 - inputs.powder_uniformity)
                    porosity += rng.normal(0, 0.003)
                    porosity = float(np.clip(porosity, 0.0, 0.20))
                    row["porosity_fraction"] = porosity
                    row["relative_density"] = 1.0 - porosity

                    for modality in ["ot", "mpm", "pbi"]:
                        rel_dir = Path("images") / modality / build_id
                        abs_dir = out / rel_dir
                        abs_dir.mkdir(parents=True, exist_ok=True)
                        rel_path = rel_dir / f"{sample_id}_{modality}.png"
                        abs_path = out / rel_path

                        img = image_from_process(
                            inputs,
                            PATCH_SIZES_HW[modality],
                            modality,
                            seed=abs(hash((sample_id, modality))) % (2**32),
                        )
                        if not isinstance(img, Image.Image):
                            img = Image.fromarray(np.asarray(img))
                        img.save(abs_path)
                        row[f"{modality}_path"] = str(rel_path)

                    rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False)
    print(f"Wrote {len(manifest)} rows to {out / 'manifest.csv'}")
    print(manifest["class_name"].value_counts())


if __name__ == "__main__":
    main()
