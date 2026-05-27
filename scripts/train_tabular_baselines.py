#!/usr/bin/env python
"""Train process-only, sensor-only, and hybrid tabular baselines.

This is the first serious training step for LayerWise-QC.

It trains ablation models on the feature table created by:

    scripts/build_feature_table.py

Supported tasks:
    classification:
        target = class_name, ct_label, metallography_label, surface_defect_label, etc.
    regression:
        target = relative_density, porosity_fraction, etc.

Recommended validation:
    group split by build_id or specimen_id when those columns exist.

Examples
--------
Classification:
python scripts/train_tabular_baselines.py \
    --features outputs/features/demo_features.csv \
    --target class_name \
    --task classification \
    --group-col specimen_id \
    --out-dir outputs/training/tabular_demo

Regression:
python scripts/train_tabular_baselines.py \
    --features outputs/features/real_features.csv \
    --target relative_density \
    --task regression \
    --group-col build_id \
    --out-dir outputs/training/rd_regression
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from am_defect_detection.feature_table import infer_feature_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Feature table CSV.")
    parser.add_argument("--target", default="class_name", help="Target column.")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--group-col", default=None, help="Prefer build_id or specimen_id to avoid leakage.")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def split_data(
    df: pd.DataFrame,
    target: str,
    group_col: str | None,
    test_size: float,
    random_state: int,
):
    """Split data, preferring group-wise split when possible."""
    y = df[target]
    valid = y.notna()
    df = df.loc[valid].reset_index(drop=True)
    y = df[target]

    if group_col and group_col in df.columns and df[group_col].notna().nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        groups = df[group_col].astype(str)
        train_idx, test_idx = next(splitter.split(df, y, groups=groups))
        split_type = f"group_split_by_{group_col}"
    else:
        stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        split_type = "random_stratified_split" if stratify is not None else "random_split"

    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy(), split_type


def numeric_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_models(task: str, random_state: int) -> dict[str, Any]:
    if task == "classification":
        return {
            "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
            "logistic_regression": LogisticRegression(max_iter=5000, class_weight="balanced"),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                class_weight="balanced",
                min_samples_leaf=2,
            ),
            "hist_gradient_boosting": HistGradientBoostingClassifier(
                random_state=random_state,
            ),
        }

    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            min_samples_leaf=2,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            random_state=random_state,
        ),
    }


def evaluate_classification(y_true, y_pred) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_regression(y_true, y_pred) -> dict[str, Any]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    if args.target not in df.columns:
        raise ValueError(f"Target column not found: {args.target}")

    groups = infer_feature_groups(df)
    train_df, test_df, split_type = split_data(
        df,
        target=args.target,
        group_col=args.group_col,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    all_metrics: dict[str, Any] = {
        "feature_table": args.features,
        "target": args.target,
        "task": args.task,
        "split_type": split_type,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "results": {},
    }

    predictions_rows = []

    for group_name, feature_cols in groups.items():
        feature_cols = [col for col in feature_cols if col in df.columns]
        if not feature_cols:
            print(f"Skipping {group_name}: no features.")
            continue

        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        y_train = train_df[args.target]
        y_test = test_df[args.target]

        models = make_models(args.task, args.random_state)

        for model_name, estimator in models.items():
            pipe = Pipeline(
                steps=[
                    ("preprocess", numeric_preprocessor(feature_cols)),
                    ("model", estimator),
                ]
            )

            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            if args.task == "classification":
                metrics = evaluate_classification(y_test, pred)
            else:
                metrics = evaluate_regression(y_test, pred)

            key = f"{group_name}__{model_name}"
            all_metrics["results"][key] = {
                "feature_group": group_name,
                "model": model_name,
                "n_features": len(feature_cols),
                "features": feature_cols,
                "metrics": metrics,
            }

            model_path = out_dir / f"{key}.joblib"
            joblib.dump(pipe, model_path)
            all_metrics["results"][key]["model_path"] = str(model_path)

            for i, (idx, truth, p) in enumerate(zip(test_df.index, y_test, pred)):
                predictions_rows.append(
                    {
                        "feature_group": group_name,
                        "model": model_name,
                        "row_index": int(idx),
                        "y_true": truth,
                        "y_pred": p,
                    }
                )

            print(f"Finished {key}")

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    pred_path = out_dir / "predictions.csv"
    pd.DataFrame(predictions_rows).to_csv(pred_path, index=False)

    # Compact leaderboard
    leaderboard_rows = []
    for key, result in all_metrics["results"].items():
        row = {
            "run": key,
            "feature_group": result["feature_group"],
            "model": result["model"],
            "n_features": result["n_features"],
        }
        metrics = result["metrics"]
        if args.task == "classification":
            row.update(
                {
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                }
            )
        else:
            row.update(
                {
                    "r2": metrics["r2"],
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                }
            )
        leaderboard_rows.append(row)

    leaderboard = pd.DataFrame(leaderboard_rows)
    if not leaderboard.empty:
        sort_col = "macro_f1" if args.task == "classification" else "r2"
        leaderboard = leaderboard.sort_values(sort_col, ascending=False)
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote leaderboard: {out_dir / 'leaderboard.csv'}")


if __name__ == "__main__":
    main()
