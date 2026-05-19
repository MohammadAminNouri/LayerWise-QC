"""Experiment constants for the layer-wise design of experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

CLASS_NAMES: List[str] = [
    "standard",
    "delta_minus_30_ved",
    "delta_plus_30_ved",
]
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for name, idx in CLASS_TO_IDX.items()}


@dataclass(frozen=True)
class ProcessCondition:
    class_name: str
    laser_power_w: float
    scan_speed_mm_s: float
    hatch_distance_mm: float = 0.12
    layer_thickness_mm: float = 0.06

    @property
    def ved_j_mm3(self) -> float:
        return calculate_ved(
            power_w=self.laser_power_w,
            scan_speed_mm_s=self.scan_speed_mm_s,
            hatch_distance_mm=self.hatch_distance_mm,
            layer_thickness_mm=self.layer_thickness_mm,
        )


def calculate_ved(
    power_w: float,
    scan_speed_mm_s: float,
    hatch_distance_mm: float,
    layer_thickness_mm: float,
) -> float:
    """Calculate volumetric energy density in J/mm^3.

    VED = P / (v * h * L)
    """
    if scan_speed_mm_s <= 0 or hatch_distance_mm <= 0 or layer_thickness_mm <= 0:
        raise ValueError("Scan speed, hatch distance, and layer thickness must be positive.")
    return power_w / (scan_speed_mm_s * hatch_distance_mm * layer_thickness_mm)


PROCESS_CONDITIONS: Dict[str, ProcessCondition] = {
    "standard": ProcessCondition("standard", laser_power_w=340, scan_speed_mm_s=1250),
    "delta_minus_30_ved": ProcessCondition("delta_minus_30_ved", laser_power_w=238, scan_speed_mm_s=1250),
    "delta_plus_30_ved": ProcessCondition("delta_plus_30_ved", laser_power_w=370, scan_speed_mm_s=1046.38),
}

# The specimen layout: 10 standard references, 10 low-VED specimens, 10 high-VED specimens.
# Within low/high rows, the number of self-induced defective layers increases from 1 to 10.
SPECIMEN_LAYOUT = []
for specimen_id in range(1, 31):
    if 1 <= specimen_id <= 10:
        exposure = "standard"
        defective_layers = specimen_id
    elif 11 <= specimen_id <= 20:
        exposure = "delta_minus_30_ved"
        defective_layers = specimen_id - 10
    else:
        exposure = "delta_plus_30_ved"
        defective_layers = specimen_id - 20
    SPECIMEN_LAYOUT.append(
        {
            "specimen_id": specimen_id,
            "exposure": exposure,
            "defective_layers": defective_layers,
        }
    )

PATCH_SIZES_HW = {
    "ot": (119, 79),
    "mpm": (152, 103),
    # Placeholder for the thesis version if powder-bed imaging replaces meltpool monitoring.
    "pbi": (152, 103),
}
