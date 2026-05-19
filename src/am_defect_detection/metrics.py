"""Evaluation metrics for imbalanced AM defect detection."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from .constants import CLASS_NAMES


def compute_metrics(y_true: List[int] | np.ndarray, y_pred: List[int] | np.ndarray) -> Dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(range(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=CLASS_NAMES,
            zero_division=0,
            output_dict=True,
        ),
    }
