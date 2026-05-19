"""Dataset and data-loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .constants import CLASS_NAMES, PATCH_SIZES_HW
from .transforms import build_transforms


@dataclass
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sample_id", "class_idx", "class_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    return df


def make_splits(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> SplitFrames:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    labels = df["class_idx"].astype(int)
    stratify = labels if labels.value_counts().min() >= 2 else None
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=stratify,
    )
    rel_test_size = test_size / (val_size + test_size)
    temp_labels = temp_df["class_idx"].astype(int)
    stratify_temp = temp_labels if temp_labels.value_counts().min() >= 2 else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test_size,
        random_state=seed,
        stratify=stratify_temp,
    )
    return SplitFrames(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


class ModalityDataset(Dataset):
    """Single-modality dataset for OT, MPM, or PBI patches."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        modality: str,
        root: str | Path | None = None,
        train: bool = True,
        patch_size_hw: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.df = manifest.reset_index(drop=True).copy()
        self.modality = modality.lower()
        self.root = Path(root) if root else None
        self.path_column = f"{self.modality}_path"
        if self.path_column not in self.df.columns:
            raise ValueError(f"Manifest must contain column '{self.path_column}'.")
        self.patch_size_hw = patch_size_hw or PATCH_SIZES_HW.get(self.modality, (224, 224))
        self.transform = build_transforms(self.patch_size_hw, train=train)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute() or self.root is None:
            return path
        return self.root / path

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int]:
        row = self.df.iloc[idx]
        path = self._resolve_path(row[self.path_column])
        image = Image.open(path).convert("RGB")
        image_tensor = self.transform(image)
        label = int(row["class_idx"])
        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": str(row["sample_id"]),
            "image_path": str(path),
        }


def build_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    labels = df["class_idx"].astype(int).to_numpy()
    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    class_counts[class_counts == 0] = 1.0
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def save_splits(splits: SplitFrames, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits.train.to_csv(out_dir / "train_split.csv", index=False)
    splits.val.to_csv(out_dir / "val_split.csv", index=False)
    splits.test.to_csv(out_dir / "test_split.csv", index=False)


def load_split_or_create(
    manifest_path: str | Path,
    out_dir: str | Path,
    seed: int = 42,
) -> SplitFrames:
    out_dir = Path(out_dir)
    train_csv = out_dir / "train_split.csv"
    val_csv = out_dir / "val_split.csv"
    test_csv = out_dir / "test_split.csv"
    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        return SplitFrames(
            train=pd.read_csv(train_csv),
            val=pd.read_csv(val_csv),
            test=pd.read_csv(test_csv),
        )
    df = read_manifest(manifest_path)
    splits = make_splits(df, seed=seed)
    save_splits(splits, out_dir)
    return splits
