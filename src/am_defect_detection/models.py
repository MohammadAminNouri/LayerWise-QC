"""Model definitions: ResNet-18 and late-fusion ensemble."""

from __future__ import annotations

import warnings
from typing import Dict, Tuple

import torch
from torch import nn


class LateFusionEnsemble(nn.Module):
    """Average probability outputs from modality-specific models."""

    def __init__(self, models_by_modality: Dict[str, nn.Module], weights: Dict[str, float] | None = None) -> None:
        super().__init__()
        self.models_by_modality = nn.ModuleDict(models_by_modality)
        if weights is None:
            weights = {name: 1.0 / len(models_by_modality) for name in models_by_modality}
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Fusion weights must sum to a positive value.")
        self.weights = {name: float(value) / total for name, value in weights.items()}

    @torch.no_grad()
    def predict_proba(self, batch_by_modality: Dict[str, torch.Tensor]) -> torch.Tensor:
        probs = None
        for name, model in self.models_by_modality.items():
            logits = model(batch_by_modality[name])
            p = torch.softmax(logits, dim=1) * self.weights[name]
            probs = p if probs is None else probs + p
        return probs

    def forward(self, batch_by_modality: Dict[str, torch.Tensor]) -> torch.Tensor:
        probs = []
        for name, model in self.models_by_modality.items():
            logits = model(batch_by_modality[name])
            probs.append(torch.softmax(logits, dim=1) * self.weights[name])
        return torch.stack(probs, dim=0).sum(dim=0)


class FallbackCNN(nn.Module):
    """Small fallback CNN used only when torchvision is unavailable/broken.

    The intended architecture is ResNet-18. This fallback exists so tests and demos
    can still run in minimal environments with incompatible torch/torchvision builds.
    """

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer1 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def build_resnet18(num_classes: int = 3, pretrained: bool = True, allow_fallback: bool = True) -> nn.Module:
    """Build a ResNet-18 classifier.

    The intended model is torchvision ResNet-18. If torchvision cannot be
    imported because of a local binary mismatch, a small fallback CNN is returned
    unless ``allow_fallback=False``.
    """
    try:
        from torchvision import models

        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        try:
            model = models.resnet18(weights=weights)
        except Exception:
            model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.model_family = "torchvision_resnet18"
    except Exception as exc:
        if not allow_fallback:
            raise
        warnings.warn(
            "torchvision ResNet-18 could not be constructed; using FallbackCNN. "
            f"Original error: {exc}",
            RuntimeWarning,
        )
        model = FallbackCNN(num_classes=num_classes)
        model.model_family = "fallback_cnn"

    for parameter in model.parameters():
        parameter.requires_grad = True
    return model


def build_model_by_family(model_family: str, num_classes: int = 3) -> nn.Module:
    if model_family == "fallback_cnn":
        return FallbackCNN(num_classes=num_classes)
    return build_resnet18(num_classes=num_classes, pretrained=False)


def load_checkpoint_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = int(checkpoint.get("num_classes", 3))
    model_family = checkpoint.get("model_family", "torchvision_resnet18")
    model = build_model_by_family(model_family, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
