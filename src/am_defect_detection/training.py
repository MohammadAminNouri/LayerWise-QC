"""Training and inference loops."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * images.size(0)
        n += images.size(0)
    return running_loss / max(n, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    probs_all: List[np.ndarray] = []
    sample_ids: List[str] = []
    for batch in tqdm(loader, desc="predict", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].cpu().numpy().astype(int)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        pred = probs.argmax(axis=1)
        y_true.extend(labels.tolist())
        y_pred.extend(pred.tolist())
        probs_all.append(probs)
        sample_ids.extend([str(x) for x in batch["sample_id"]])
    return np.asarray(y_true), np.asarray(y_pred), np.vstack(probs_all), sample_ids


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    y_true, y_pred, _, _ = predict(model, loader, device)
    return compute_metrics(y_true, y_pred)


def moving_average(values: List[float], window: int = 3) -> float:
    if not values:
        return float("-inf")
    window = max(1, min(window, len(values)))
    return float(np.mean(values[-window:]))


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    modality: str,
    metrics: Dict,
    num_classes: int = 3,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "modality": modality,
            "metrics": metrics,
            "num_classes": num_classes,
            "model_family": getattr(model, "model_family", "torchvision_resnet18"),
        },
        path,
    )
