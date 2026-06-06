"""Advanced CatBoost model training for thesis experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier, CatBoostRegressor
import pandas as pd
import numpy as np

from aac_adoption.config import RANDOM_STATE
from aac_adoption.features.feature_sets import (
    INTAKE_TIME_FEATURES,
    available_features_for_df,
    model_feature_columns,
)
from aac_adoption.models.artifacts import save_model_artifact
from aac_adoption.models.evaluate import classification_metrics, regression_metrics
from aac_adoption.models.split import DatasetSplit, make_time_split
from aac_adoption.models.train_baseline import ANIMAL_SUBSETS, CATEGORICAL_FEATURES, limit_rows
from aac_adoption.models.metadata import base_training_metadata
from aac_adoption.analysis.survival_analysis import log_transform_LOS, compute_kaplan_meier_survival
from aac_adoption.features.feature_engineering import winsorize_outliers


@dataclass(frozen=True)
class AdvancedTrainingOutputs:
    """Metric tables returned by advanced training."""

    classification_metrics: pd.DataFrame
    regression_metrics: pd.DataFrame


def categorical_features_for(feature_columns: list[str]) -> list[str]:
    """Return categorical feature names for CatBoost."""
    configured = set(CATEGORICAL_FEATURES + ["is_mixed_breed", "intake_condition", "sex_upon_intake"])
    return [column for column in feature_columns if column in configured]


def prepare_catboost_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Prepare feature frame with stable missing-value handling for CatBoost."""
    result = df[feature_columns].copy()
    categorical_features = categorical_features_for(feature_columns)
    for column in categorical_features:
        result[column] = result[column].astype("string").fillna("unknown").astype(str)
    for column in feature_columns:
        if column not in categorical_features:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _fit_and_save(
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
    fit_kwargs: dict[str, Any] = {
        "X": train_x,
        "y": split.train[target_column],
        "cat_features": categorical_features,
        "verbose": False,
    }
    if validation_x is not None:
        fit_kwargs["eval_set"] = (validation_x, split.validation[target_column])
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
    metadata["feature_columns"] = feature_columns
    path = save_model_artifact(model, models_dir, task, split.animal_subset, "catboost", metadata)
    metadata["artifact_path"] = str(path)
    return model, metadata


def train_advanced_classification(
    df: pd.DataFrame,
    models_dir: Path,
    run_timestamp: str,
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    early_stopping_rounds: int,
) -> list[dict[str, Any]]:
    """Train CatBoost adoption classifiers."""
    rows: list[dict[str, Any]] = []
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "early_stopping_rounds": early_stopping_rounds,
        "random_seed": RANDOM_STATE,
    }
    for subset in ANIMAL_SUBSETS:
        split = make_time_split(df, "classification_target", animal_subset=subset)
        feature_columns = model_feature_columns(split.train)
        model, metadata = _fit_and_save(
            model=CatBoostClassifier(**params),
            task="classification",
            split=split,
            feature_columns=feature_columns,
            target_column="classification_target",
            models_dir=models_dir,
            run_timestamp=run_timestamp,
            params=params,
        )
        test_x = prepare_catboost_frame(split.test, feature_columns)
        predictions = model.predict(test_x).astype(int)
        scores = model.predict_proba(test_x)[:, 1]
        metrics = classification_metrics(
            split.test["classification_target"], 
            predictions, 
            scores,
            compute_ci=(subset in ["dogs", "cats"])
        )
        rows.append({**metadata, **metrics})
    return rows


def train_advanced_regression(
    df: pd.DataFrame,
    models_dir: Path,
    run_timestamp: str,
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    early_stopping_rounds: int,
) -> list[dict[str, Any]]:
    """Train CatBoost days-to-outcome regressors with log-transform and adopted-only filtering."""
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
    for subset in ANIMAL_SUBSETS:
        split = make_time_split(df, "regression_target_days", animal_subset=subset)
        feature_columns = model_feature_columns(split.train)
        
        filter_df = split.train.copy()
        filter_df = filter_df[filter_df["adopted"]].copy()
        filter_df = log_transform_LOS(filter_df, "regression_target_days")
        filter_df = filter_df.copy()
        
        split_train = split.train.copy()
        split_train["regression_target_days"] = filter_df["log_regression_target_days"]
        
        model, metadata = _fit_and_save(
            model=CatBoostRegressor(**params),
            task="regression",
            split=DatasetSplit(
                full_data=split.full_data,
                train=split_train,
                validation=split.validation,
                test=split.test,
                strategy=split.strategy,
                train_period=split.train_period,
                validation_period=split.validation_period,
                test_period=split.test_period,
                animal_subset=split.animal_subset,
            ),
            feature_columns=feature_columns,
            target_column="log_regression_target_days",
            models_dir=models_dir,
            run_timestamp=run_timestamp,
            params=params,
        )
        test_x = prepare_catboost_frame(split.test, feature_columns)
        predictions = model.predict(test_x)
        predictions_exp = np.exp(predictions) - 1
        rows.append({**metadata, **regression_metrics(split.test["regression_target_days"], predictions_exp)})
    return rows


def train_all_advanced(
    data_path: str | Path,
    metrics_dir: str | Path = "reports/metrics",
    models_dir: str | Path = "models/advanced",
    max_rows: int | None = None,
    iterations: int = 1000,
    learning_rate: float = 0.05,
    depth: int = 6,
    early_stopping_rounds: int = 50,
    tuned_params_path: str | Path | None = None,
) -> AdvancedTrainingOutputs:
    """Train all advanced CatBoost models and save metric outputs."""
    header = pd.read_csv(data_path, nrows=0)
    parse_dates = [col for col in ["intake_datetime", "outcome_datetime"] if col in header.columns]
    df = pd.read_csv(data_path, parse_dates=parse_dates)
    df = limit_rows(df, max_rows)

    metrics_output_dir = Path(metrics_dir)
    model_output_dir = Path(models_dir)
    metrics_output_dir.mkdir(parents=True, exist_ok=True)
    
    clf_kwargs = {"iterations": iterations, "learning_rate": learning_rate, "depth": depth}
    reg_kwargs = {"iterations": iterations, "learning_rate": learning_rate, "depth": depth}
    
    if tuned_params_path is not None:
        import json
        p = Path(tuned_params_path)
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            if "catboost_classification" in d:
                clf_kwargs.update(d["catboost_classification"])
            if "catboost_regression" in d:
                reg_kwargs.update(d["catboost_regression"])

    run_timestamp = datetime.now(timezone.utc).isoformat()
    classification = pd.DataFrame(
        train_advanced_classification(
            df,
            model_output_dir,
            run_timestamp,
            early_stopping_rounds=early_stopping_rounds,
            **clf_kwargs
        )
    )
    regression = pd.DataFrame(
        train_advanced_regression(
            df,
            model_output_dir,
            run_timestamp,
            early_stopping_rounds=early_stopping_rounds,
            **reg_kwargs
        )
    )

    classification.to_csv(metrics_output_dir / "advanced_classification_metrics.csv", index=False)
    regression.to_csv(metrics_output_dir / "advanced_regression_metrics.csv", index=False)
    pd.concat([classification, regression], ignore_index=True, sort=False).to_csv(
        metrics_output_dir / "advanced_metrics.csv",
        index=False,
    )
    return AdvancedTrainingOutputs(classification_metrics=classification, regression_metrics=regression)
