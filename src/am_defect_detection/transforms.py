"""Image transforms matching the augmentation ideas used in the paper."""

from __future__ import annotations

import random
from typing import Callable, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch


class AddGaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 0.2):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


def _pil_to_tensor_normalized(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


class FallbackTransform:
    """Torchvision-free transform used when torchvision is unavailable."""

    def __init__(self, size_hw: Tuple[int, int], train: bool = True) -> None:
        self.size_hw = size_hw
        self.train = train
        self.noise = AddGaussianNoise(mean=0.0, std=0.2)

    def _random_resized_crop(self, img: Image.Image) -> Image.Image:
        h, w = self.size_hw
        width, height = img.size
        scale = random.uniform(0.85, 1.0)
        crop_w = max(1, int(width * scale))
        crop_h = max(1, int(height * scale))
        left = random.randint(0, max(0, width - crop_w))
        top = random.randint(0, max(0, height - crop_h))
        img = img.crop((left, top, left + crop_w, top + crop_h))
        return img.resize((w, h), Image.BILINEAR)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        h, w = self.size_hw
        img = img.convert("RGB")
        if self.train:
            img = self._random_resized_crop(img)
            if random.random() < 0.5:
                img = ImageOps.mirror(img)
            if random.random() < 0.5:
                img = ImageOps.flip(img)
            if random.random() < 0.5:
                img = img.rotate(180)
            if random.random() < 0.25:
                img = ImageEnhance.Sharpness(img).enhance(0.2)
            if random.random() < 0.25:
                img = ImageEnhance.Sharpness(img).enhance(2.0)
            if random.random() < 0.5:
                img = ImageOps.autocontrast(img)
            # Approximate brightness/saturation jitter.
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))
            img = ImageEnhance.Color(img).enhance(random.uniform(0.5, 1.5))
            if random.random() < 0.5:
                img = ImageOps.equalize(img)
            if random.random() < 0.5:
                img = ImageOps.invert(img)
            tensor = _pil_to_tensor_normalized(img)
            if random.random() < 0.5:
                # Noise should be applied before normalization ideally; here it is
                # applied after normalization as a robust perturbation.
                tensor = tensor + torch.randn_like(tensor) * 0.2
            return tensor
        img = img.resize((w, h), Image.BILINEAR)
        return _pil_to_tensor_normalized(img)


def build_transforms(size_hw: Tuple[int, int], train: bool = True) -> Callable[[Image.Image], torch.Tensor]:
    """Build transforms for one modality.

    Uses torchvision when available; otherwise falls back to PIL/torch transforms.
    """
    try:
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        h, w = size_hw
        if train:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=(h, w), scale=(0.85, 1.0), interpolation=InterpolationMode.BILINEAR),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomApply([transforms.Lambda(lambda img: img.rotate(180))], p=0.5),
                    transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=0.2)], p=0.25),
                    transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2.0)], p=0.25),
                    transforms.RandomAutocontrast(p=0.5),
                    transforms.ColorJitter(brightness=0.5, saturation=0.5, hue=0.3),
                    transforms.RandomEqualize(p=0.5),
                    transforms.RandomInvert(p=0.5),
                    transforms.ToTensor(),
                    transforms.RandomApply([AddGaussianNoise(mean=0.0, std=0.2)], p=0.5),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize(size=(h, w), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    except Exception:
        return FallbackTransform(size_hw, train=train)
