"""Train CatBoost regressors strictly for adopted animals (days-to-adoption)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from aac_adoption.config import RANDOM_STATE
from aac_adoption.features.feature_sets import model_feature_columns
from aac_adoption.models.artifacts import save_model_artifact
from aac_adoption.models.evaluate import regression_metrics
from aac_adoption.models.split import DatasetSplit, make_time_split
from aac_adoption.models.train_baseline import ANIMAL_SUBSETS, limit_rows
from aac_adoption.models.train_advanced import (
    categorical_features_for,
    prepare_catboost_frame,
)
from aac_adoption.models.metadata import base_training_metadata


@dataclass(frozen=True)
class AdoptedRegressionOutputs:
    """Metric tables returned by adopted-only regression training."""

    regression_metrics: pd.DataFrame


def _fit_and_save_adopted(
    *,
    model,
    task: str,
    split: DatasetSplit,
    feature_columns: list[str],
    target_column: str,
    models_dir: Path,
    run_timestamp: str,
    params: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    categorical_features = categorical_features_for(feature_columns)
    train_x = prepare_catboost_frame(split.train, feature_columns)
    validation_x = prepare_catboost_frame(split.validation, feature_columns) if not split.validation.empty else None
    
    train_y = np.log1p(split.train[target_column])
    validation_y = np.log1p(split.validation[target_column]) if not split.validation.empty else None

    fit_kwargs: dict[str, Any] = {
        "X": train_x,
        "y": train_y,
        "cat_features": categorical_features,
        "verbose": False,
    }
    if validation_x is not None:
        fit_kwargs["eval_set"] = (validation_x, validation_y)
        fit_kwargs["use_best_model"] = True
    model.fit(**fit_kwargs)

    metadata = base_training_metadata(
        model_name="catboost",
        task=task,
        split=split,
        feature_columns=feature_columns,
        run_timestamp=run_timestamp,
        categorical_features=categorical_features,
        params=params,
    )
    metadata["target_transform"] = "log1p"
    metadata["training_target_min"] = float(train_y.min())
    metadata["training_target_max"] = float(train_y.max())
    path = save_model_artifact(model, models_dir, task, split.animal_subset, "catboost", metadata)
    metadata["artifact_path"] = str(path)
    return model, metadata


def train_adopted_regression(
    df: pd.DataFrame,
    models_dir: Path,
    run_timestamp: str,
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    early_stopping_rounds: int,
) -> list[dict[str, Any]]:
    """Train CatBoost days-to-adoption regressors on adopted subset."""
    rows: list[dict[str, Any]] = []
    params = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "early_stopping_rounds": early_stopping_rounds,
        "random_seed": RANDOM_STATE,
    }
    
    # Filter to adopted only
    adopted_df = df.loc[
        df["classification_target"].eq(1) & df["days_to_adoption"].notna()
    ].copy()

    for subset in ANIMAL_SUBSETS:
        split = make_time_split(adopted_df, "days_to_adoption", animal_subset=subset)
        feature_columns = model_feature_columns(split.train)
        model, metadata = _fit_and_save_adopted(
            model=CatBoostRegressor(**params),
            task="regression_adopted",
            split=split,
            feature_columns=feature_columns,
            target_column="days_to_adoption",
            models_dir=models_dir,
            run_timestamp=run_timestamp,
            params=params,
        )
        
        if not split.selection.empty:
            sel_x = prepare_catboost_frame(split.selection, feature_columns)
            log_predictions = model.predict(sel_x)
            predictions = np.expm1(log_predictions)
            predictions = np.maximum(predictions, 0.0)
            sel_metrics = regression_metrics(split.selection["days_to_adoption"], predictions)
            rows.append({
                **metadata,
                **sel_metrics,
                "target_column": "days_to_adoption",
                "target_transform": "log1p",
                "prediction_inverse_transform": "expm1",
                "metric_split": "selection",
                "selection_eligible": 1,
            })
            
        if not split.test.empty:
            test_x = prepare_catboost_frame(split.test, feature_columns)
            log_predictions = model.predict(test_x)
            predictions = np.expm1(log_predictions)
            predictions = np.maximum(predictions, 0.0)
            metrics = regression_metrics(split.test["days_to_adoption"], predictions)
            rows.append({
                **metadata,
                **metrics,
                "target_column": "days_to_adoption",
                "target_transform": "log1p",
                "prediction_inverse_transform": "expm1",
                "metric_split": "test",
                "selection_eligible": 0,
            })
    return rows


def train_all_adopted(
    data_path: str | Path,
    metrics_dir: str | Path = "reports/metrics",
    models_dir: str | Path = "models/advanced",
    max_rows: int | None = None,
    iterations: int = 1000,
    learning_rate: float = 0.05,
    depth: int = 6,
    early_stopping_rounds: int = 50,
) -> AdoptedRegressionOutputs:
    """Train all adopted-only CatBoost regressors and save metric outputs."""
    header = pd.read_csv(data_path, nrows=0)
    parse_dates = [col for col in ["intake_datetime", "outcome_datetime"] if col in header.columns]
    df = pd.read_csv(data_path, parse_dates=parse_dates)
    df = limit_rows(df, max_rows)

    metrics_output_dir = Path(metrics_dir)
    model_output_dir = Path(models_dir)
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now(timezone.utc).isoformat()
    regression = pd.DataFrame(
        train_adopted_regression(
            df,
            model_output_dir,
            run_timestamp,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            early_stopping_rounds=early_stopping_rounds,
        )
    )

    regression.to_csv(metrics_output_dir / "adopted_regression_metrics.csv", index=False)
    return AdoptedRegressionOutputs(regression_metrics=regression)
