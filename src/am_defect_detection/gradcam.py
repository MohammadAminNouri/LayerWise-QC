"""Minimal Grad-CAM implementation for ResNet-like models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module, _input, output):
            self.activations = output.detach()

        def backward_hook(_module, _grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, image_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients/activations.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_np -= cam_np.min()
        cam_np /= cam_np.max() + 1e-8
        return cam_np, class_idx


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    img = tensor.squeeze(0) * std + mean
    img = torch.clamp(img, 0, 1).permute(1, 2, 0).detach().cpu().numpy()
    return (img * 255).astype(np.uint8)


def save_gradcam_overlay(image_tensor: torch.Tensor, cam: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = denormalize_image(image_tensor)
    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * base + 0.45 * heatmap).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)
