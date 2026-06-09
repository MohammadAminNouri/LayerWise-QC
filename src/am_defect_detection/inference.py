"""Inference helpers for trained LayerWise-QC image models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
import torch

from .constants import CLASS_NAMES, PATCH_SIZES_HW
from .models import load_checkpoint_model
from .transforms import build_transforms


@dataclass
class ModelBundle:
    modality: str
    checkpoint_path: Path
    class_names: list[str]
    model: torch.nn.Module
    transform: Callable
    metadata: dict
    device: torch.device


def load_model_bundle(checkpoint_path: str | Path, modality: str, device: str = "cpu") -> ModelBundle:
    """Load a trained modality checkpoint with its preprocessing transform."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    torch_device = torch.device(device)
    model, metadata = load_checkpoint_model(str(checkpoint_path), torch_device)
    modality = modality.lower()
    patch_size = PATCH_SIZES_HW.get(modality, (224, 224))
    transform = build_transforms(patch_size, train=False)
    class_names = metadata.get("class_names", CLASS_NAMES)
    return ModelBundle(modality, checkpoint_path, list(class_names), model, transform, metadata, torch_device)


def _to_pil(image: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


@torch.no_grad()
def predict_image_bundle(bundle: ModelBundle, image: np.ndarray | Image.Image) -> dict:
    """Predict one image and return class probabilities and confidence."""
    pil = _to_pil(image)
    tensor = bundle.transform(pil).unsqueeze(0).to(bundle.device)
    logits = bundle.model(tensor)
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    class_scores = {name: float(probs[i]) for i, name in enumerate(bundle.class_names[: len(probs)])}
    best_idx = int(np.argmax(probs))
    return {
        "class_scores": class_scores,
        "predicted_class": bundle.class_names[best_idx],
        "confidence": float(probs[best_idx]),
        "modality": bundle.modality,
        "checkpoint_path": str(bundle.checkpoint_path),
    }
