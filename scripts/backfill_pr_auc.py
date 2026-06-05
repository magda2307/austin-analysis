"""Backfill PR-AUC metrics for existing classification models."""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aac_adoption.models.split import make_time_split
from aac_adoption.models.train_baseline import feature_columns_for
from aac_adoption.models.train_advanced import prepare_catboost_frame


def backfill_file(csv_path: Path, df_base: pd.DataFrame, df_context: pd.DataFrame) -> None:
    if not csv_path.exists():
        print(f"File {csv_path} does not exist. Skipping.")
        return

    df_metrics = pd.read_csv(csv_path)
    if "pr_auc" not in df_metrics.columns:
        df_metrics["pr_auc"] = None

    updated = False
    for idx, row in df_metrics.iterrows():
        if row.get("task") != "classification":
            continue

        artifact_path_str = row.get("artifact_path")
        if pd.isna(artifact_path_str):
            continue

        artifact_path = PROJECT_ROOT / artifact_path_str
        if not artifact_path.exists():
            print(f"Artifact {artifact_path} not found. Skipping row {idx}.")
            continue

        # Choose the right dataset based on path
        is_context = "context" in str(artifact_path).lower()
        df_all = df_context if is_context else df_base

        subset = row["animal_subset"]
        print(f"Loading {artifact_path} (is_context={is_context}) for subset {subset}...")
        model = joblib.load(artifact_path)

        # Split data
        split = make_time_split(df_all, "classification_target", animal_subset=subset)

        # Inspect feature names from model
        if hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)
        elif hasattr(model, "feature_names_"):
            features = list(model.feature_names_)
        elif hasattr(model, "named_steps") and "preprocess" in model.named_steps:
            preprocess = model.named_steps["preprocess"]
            if hasattr(preprocess, "feature_names_in_"):
                features = list(preprocess.feature_names_in_)
            else:
                features = feature_columns_for(split.train)
        else:
            features = feature_columns_for(split.train)

        y_test = split.test["classification_target"]

        # Run predictions
        model_name = row["model_name"]
        if "catboost" in model_name.lower():
            X_test = prepare_catboost_frame(split.test, features)
        else:
            X_test = split.test[features]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            pr_auc = float(average_precision_score(y_test, probs))
            df_metrics.at[idx, "pr_auc"] = pr_auc
            print(f"  Calculated PR-AUC: {pr_auc:.4f}")
            updated = True
        else:
            print(f"  Model {model_name} has no predict_proba method.")

    if updated:
        df_metrics.to_csv(csv_path, index=False)
        print(f"Successfully updated {csv_path}")


def main() -> None:
    data_path = PROJECT_ROOT / "data/processed/modeling_dataset.csv"
    context_data_path = PROJECT_ROOT / "data/processed/modeling_dataset_context.csv"

    if not data_path.exists():
        print(f"Data path {data_path} not found.")
        sys.exit(1)

    print("Reading base modeling dataset...")
    df_base = pd.read_csv(data_path)

    df_context = None
    if context_data_path.exists():
        print("Reading context modeling dataset...")
        df_context = pd.read_csv(context_data_path)
    else:
        print("Context modeling dataset not found, using base dataset for all.")
        df_context = df_base

    metrics_dir = PROJECT_ROOT / "reports/metrics"
    files_to_update = [
        "classification_metrics.csv",
        "boosting_classification_metrics.csv",
        "advanced_classification_metrics.csv",
        "baseline_metrics.csv",
        "boosting_metrics.csv",
        "advanced_metrics.csv",
    ]

    for fname in files_to_update:
        backfill_file(metrics_dir / fname, df_base, df_context)


if __name__ == "__main__":
    main()
