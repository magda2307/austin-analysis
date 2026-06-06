"""Calibration of trained classification models using validation data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from aac_adoption.models.artifacts import artifact_path, save_model_artifact
from aac_adoption.models.evaluate import classification_metrics
from aac_adoption.models.split import make_time_split
from aac_adoption.models.train_baseline import ANIMAL_SUBSETS, limit_rows
from aac_adoption.models.train_advanced import prepare_catboost_frame
from aac_adoption.models.metadata import base_training_metadata


@dataclass(frozen=True)
class CalibratedOutputs:
    """Metrics for calibrated models."""

    classification_metrics: pd.DataFrame


def calibrate_classifiers(
    data_path: str | Path,
    source_artifacts: list[tuple[str, str]],  # List of (models_dir, model_name)
    metrics_dir: str | Path = "reports/metrics",
    models_dir: str | Path = "models/calibrated",
    max_rows: int | None = None,
) -> CalibratedOutputs:
    """Train calibration layers for trained classifiers."""
    header = pd.read_csv(data_path, nrows=0)
    parse_dates = [col for col in ["intake_datetime", "outcome_datetime"] if col in header.columns]
    df = pd.read_csv(data_path, parse_dates=parse_dates)
    df = limit_rows(df, max_rows)

    metrics_output_dir = Path(metrics_dir)
    model_output_dir = Path(models_dir)
    metrics_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []

    for source_dir, model_name in source_artifacts:
        for subset in ANIMAL_SUBSETS:
            split = make_time_split(df, "classification_target", animal_subset=subset)
            if split.validation.empty or split.test.empty:
                continue
                
            source_path = artifact_path(source_dir, "classification", subset, model_name)
            if not source_path.exists():
                print(f"Skipping missing artifact: {source_path}")
                continue
                
            model = joblib.load(source_path)
            
            import json
            metadata_path = source_path.with_suffix(".json")
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                feature_columns = metadata.get("feature_columns", list(split.validation.columns))
            else:
                feature_columns = list(split.validation.columns)
                metadata = {}

            if model_name == "catboost":
                calib_x = prepare_catboost_frame(split.validation, feature_columns)
                test_x = prepare_catboost_frame(split.test, feature_columns)
            else:
                calib_x = split.validation[feature_columns]
                test_x = split.test[feature_columns]

            from sklearn.frozen import FrozenEstimator
            calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(model), method="isotonic")
            
            print(f"[{model_name} {subset}] calib_x shape: {calib_x.shape}, nulls: {calib_x.isnull().sum().sum()}")
            if model_name == "catboost":
                cat_cols = [c for c in calib_x.columns if calib_x[c].dtype == object or calib_x[c].dtype.name == 'string']
                for c in cat_cols:
                    if calib_x[c].isnull().any():
                        print(f"NULL in cat col {c}")
                    
            calibrated.fit(calib_x, split.validation["classification_target"])
            
            predictions = calibrated.predict(test_x).astype(int)
            scores = calibrated.predict_proba(test_x)[:, 1]
            metrics = classification_metrics(split.test["classification_target"], predictions, scores)
            
            new_metadata = base_training_metadata(
                model_name=f"{model_name}_calibrated",
                task="classification_calibrated",
                split=split,
                feature_columns=feature_columns,
                run_timestamp=run_timestamp,
            )
            new_metadata["base_model_path"] = str(source_path)
            if "categorical_features" in metadata:
                new_metadata["categorical_features"] = metadata["categorical_features"]
                
            path = save_model_artifact(
                calibrated, 
                model_output_dir, 
                "classification_calibrated", 
                subset, 
                f"{model_name}_calibrated", 
                new_metadata
            )
            new_metadata["artifact_path"] = str(path)
            
            rows.append({**new_metadata, **metrics})

    output_df = pd.DataFrame(rows)
    if not output_df.empty:
        output_df.to_csv(metrics_output_dir / "calibrated_classification_metrics.csv", index=False)
        
    return CalibratedOutputs(classification_metrics=output_df)
