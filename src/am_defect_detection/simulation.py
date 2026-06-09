"""Small image and process simulation helpers used by the demo data and app.

The images are not experimental measurements. They are deliberately simple
sensor-like patterns so the pipeline can be opened, tested, and discussed before
real OT/MPM/PBI files are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .constants import CLASS_NAMES, CLASS_TO_IDX, REFERENCE_SPOT_SIZE_UM


@dataclass(frozen=True)
class ProcessInputs:
    laser_power_w: float
    scan_speed_mm_s: float
    hatch_distance_mm: float = 0.12
    layer_thickness_mm: float = 0.06
    heat_memory: float = 0.35
    powder_uniformity: float = 0.82
    spot_size_um: float = REFERENCE_SPOT_SIZE_UM

    @property
    def ved(self) -> float:
        if self.scan_speed_mm_s <= 0 or self.hatch_distance_mm <= 0 or self.layer_thickness_mm <= 0:
            raise ValueError("Scan speed, hatch distance, and layer thickness must be positive.")
        return self.laser_power_w / (self.scan_speed_mm_s * self.hatch_distance_mm * self.layer_thickness_mm)

    @property
    def spot_size_mm(self) -> float:
        if self.spot_size_um <= 0:
            raise ValueError("Spot size must be positive.")
        return self.spot_size_um / 1000.0


def classify_from_ved(ved: float, standard_ved: float = 37.78, dead_band: float = 0.18) -> str:
    """Rough label from VED. Used only by the demo app.

    A narrow band around the standard condition is kept as `standard`.
    Lower energy is mapped to lack-of-fusion risk; higher energy to keyhole risk.
    """
    ratio = ved / standard_ved
    if ratio < 1.0 - dead_band:
        return "delta_minus_30_ved"
    if ratio > 1.0 + dead_band:
        return "delta_plus_30_ved"
    return "standard"


def soft_process_scores(inputs: ProcessInputs, standard_ved: float = 37.78) -> Dict[str, float]:
    """A transparent physics proxy for the live app.

    It is not a trained model. It turns VED distance, heat memory and powder
    uniformity into class-like scores so the user can see how the logic moves.
    """
    ratio = inputs.ved / standard_ved
    spot_size_mm = max(inputs.spot_size_um / 1000.0, 1e-9)
    beam_area_mm2 = np.pi * (spot_size_mm / 2.0) ** 2
    power_density_w_mm2 = inputs.laser_power_w / beam_area_mm2
    reference_power_density = 340.0 / (np.pi * (0.08 / 2.0) ** 2)
    power_density_ratio = power_density_w_mm2 / reference_power_density
    hatch_to_spot_ratio = inputs.hatch_distance_mm / spot_size_mm

    # Spot size is deliberately a gentle correction, not a validated melt-pool model.
    # Large hatch/spot ratio weakens track overlap and raises lack-of-fusion risk.
    # Small spot/high power density raises keyhole-spatter risk even at similar VED.
    overlap_penalty = max(0.0, hatch_to_spot_ratio - 1.65) * 0.35
    broad_low_power_penalty = max(0.0, 1.0 - power_density_ratio) * max(0.0, spot_size_mm / 0.08 - 1.0) * 0.35
    concentrated_power_penalty = max(0.0, power_density_ratio - 1.0) * 0.25

    low = (
        max(0.0, 1.0 - ratio) * 2.4
        + max(0.0, 0.72 - inputs.powder_uniformity) * 1.4
        + overlap_penalty
        + broad_low_power_penalty
    )
    high = (
        max(0.0, ratio - 1.0) * 2.1
        + max(0.0, inputs.heat_memory - 0.62) * 0.9
        + concentrated_power_penalty
    )
    std = 1.15 - abs(ratio - 1.0) * 1.7 + (inputs.powder_uniformity - 0.5) * 0.25
    std -= 0.12 * max(0.0, abs(power_density_ratio - 1.0) - 0.25)
    raw = np.array([std, low, high], dtype=np.float64)
    raw = np.exp(raw - raw.max())
    probs = raw / raw.sum()
    return {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}


def fuse_scores(a: Dict[str, float], b: Dict[str, float], w_a: float = 0.5, w_b: float = 0.5) -> Dict[str, float]:
    total = max(w_a + w_b, 1e-9)
    w_a, w_b = w_a / total, w_b / total
    out = {name: w_a * float(a.get(name, 0.0)) + w_b * float(b.get(name, 0.0)) for name in CLASS_NAMES}
    s = sum(out.values()) or 1.0
    return {k: v / s for k, v in out.items()}


def _base_field(size_hw: Tuple[int, int], rng: np.random.Generator, mean: float, noise: float) -> np.ndarray:
    h, w = size_hw
    base = rng.normal(loc=mean, scale=noise, size=(h, w)).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    for _ in range(5):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        amp = rng.uniform(6, 24)
        sx, sy = rng.uniform(8, 30), rng.uniform(8, 30)
        base += amp * np.exp(-(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2)))
    return base.clip(0, 255)


def _to_rgb(field: np.ndarray) -> Image.Image:
    arr = field.clip(0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([arr, arr, arr]), mode="RGB")


def draw_standard(size_hw: tuple[int, int], rng: np.random.Generator, modality: str) -> Image.Image:
    if modality == "pbi":
        return draw_pbi(size_hw, rng, "standard")
    mean = 112 if modality == "ot" else 98
    img = _to_rgb(_base_field(size_hw, rng, mean=mean, noise=12))
    return img.filter(ImageFilter.GaussianBlur(radius=0.4))


def draw_low_ved_lack_of_fusion(img: Image.Image, rng: np.random.Generator, modality: str) -> Image.Image:
    if modality == "pbi":
        return draw_pbi((img.height, img.width), rng, "delta_minus_30_ved")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    n = 2 if modality == "ot" else 4
    for _ in range(n):
        x0 = int(rng.integers(0, max(1, w - 20)))
        y0 = int(rng.integers(0, max(1, h - 12)))
        x1 = min(w, x0 + int(rng.integers(12, 35)))
        y1 = min(h, y0 + int(rng.integers(5, 18)))
        pts = [
            (x0, y0),
            (x1, y0 + int(rng.integers(-4, 4))),
            (x1 - int(rng.integers(0, 8)), y1),
            (x0 + int(rng.integers(0, 8)), y1 + int(rng.integers(-3, 3))),
        ]
        draw.polygon(pts, fill=(24, 24, 24))
    return img.filter(ImageFilter.GaussianBlur(radius=0.2))


def draw_high_ved_keyhole_risk(img: Image.Image, rng: np.random.Generator, modality: str) -> Image.Image:
    if modality == "pbi":
        return draw_pbi((img.height, img.width), rng, "delta_plus_30_ved")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    if modality == "ot":
        overlay = Image.new("RGB", (w, h), (0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for _ in range(3):
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            r = int(rng.integers(15, max(16, min(w, h) // 2)))
            od.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(82, 82, 82))
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=12))
        arr = np.asarray(img).astype(np.int16) + np.asarray(overlay).astype(np.int16)
        return Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    for _ in range(5):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(2, 7))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(244, 244, 244))
    return img.filter(ImageFilter.GaussianBlur(radius=0.3))


def draw_pbi(size_hw: tuple[int, int], rng: np.random.Generator, class_name: str) -> Image.Image:
    """Simulate powder-bed camera patches: texture, streaks, spatters, gaps."""
    h, w = size_hw
    base = rng.normal(132, 20, size=(h, w)).astype(np.float32)
    # recoater direction texture
    for y in range(0, h, max(4, h // 18)):
        base[y : y + 1, :] += rng.uniform(-20, 20)
    img = _to_rgb(base).filter(ImageFilter.GaussianBlur(radius=0.35))
    draw = ImageDraw.Draw(img)
    if class_name == "delta_minus_30_ved":
        # local shortage / trench-like patches
        for _ in range(4):
            x0 = int(rng.integers(0, max(1, w - 25)))
            y0 = int(rng.integers(0, max(1, h - 12)))
            draw.rounded_rectangle((x0, y0, min(w, x0 + int(rng.integers(16, 40))), min(h, y0 + int(rng.integers(4, 12)))), radius=2, fill=(55, 55, 55))
    elif class_name == "delta_plus_30_ved":
        # spatter / shiny particles after unstable melting
        for _ in range(16):
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            r = int(rng.integers(1, 4))
            val = int(rng.integers(195, 250))
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(val, val, val))
    else:
        # a couple harmless grains; not empty because real images are never clean
        for _ in range(5):
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            draw.point((cx, cy), fill=(180, 180, 180))
    return img


def generate_patch(class_name: str, size_hw: tuple[int, int], rng: np.random.Generator, modality: str) -> Image.Image:
    if modality == "pbi":
        return draw_pbi(size_hw, rng, class_name)
    img = draw_standard(size_hw, rng, modality)
    if class_name == "delta_minus_30_ved":
        return draw_low_ved_lack_of_fusion(img, rng, modality)
    if class_name == "delta_plus_30_ved":
        return draw_high_ved_keyhole_risk(img, rng, modality)
    return img


def image_from_process(inputs: ProcessInputs, size_hw: Tuple[int, int], modality: str, seed: int = 11) -> Image.Image:
    rng = np.random.default_rng(seed)
    class_name = classify_from_ved(inputs.ved)
    img = generate_patch(class_name, size_hw, rng, modality)
    # Make the sliders visibly matter: heat increases brightness, poor powder uniformity adds streaks.
    arr = np.asarray(img).astype(np.float32)
    arr += (inputs.heat_memory - 0.5) * 28.0
    if modality == "pbi":
        h, w = size_hw
        for y in range(0, h, max(5, h // 12)):
            arr[y : y + 1, :, :] -= (1.0 - inputs.powder_uniformity) * 80.0
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))
