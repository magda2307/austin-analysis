"""Advanced CatBoost model training for thesis experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier, CatBoostRegressor
import pandas as pd

from aac_adoption.config import RANDOM_STATE
from aac_adoption.features.feature_sets import INTAKE_TIME_FEATURES, available_intake_features, validate_no_leakage
from aac_adoption.models.artifacts import save_model_artifact
from aac_adoption.models.evaluate import classification_metrics, regression_metrics
from aac_adoption.models.split import DatasetSplit, make_time_split
from aac_adoption.models.train_baseline import ANIMAL_SUBSETS, CATEGORICAL_FEATURES, limit_rows


@dataclass(frozen=True)
class AdvancedTrainingOutputs:
    """Metric tables returned by advanced training."""

    classification_metrics: pd.DataFrame
    regression_metrics: pd.DataFrame


def _available_features(df: pd.DataFrame, columns: list[str]) -> list[str]:
    features = [column for column in columns if column in df.columns]
    validate_no_leakage(features)
    return features


def feature_columns_for(df: pd.DataFrame) -> list[str]:
    """Return available intake-time model features."""
    features = available_intake_features(_available_features(df, INTAKE_TIME_FEATURES))
    validate_no_leakage(features)
    return features


def categorical_features_for(feature_columns: list[str]) -> list[str]:
    """Return categorical feature names for CatBoost."""
    configured = set(CATEGORICAL_FEATURES + ["age_upon_intake", "breed", "color", "has_name", "is_mixed_breed"])
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


def _base_metadata(
    *,
    model_name: str,
    task: str,
    split: DatasetSplit,
    feature_columns: list[str],
    categorical_features: list[str],
    run_timestamp: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "task": task,
        "animal_subset": split.animal_subset,
        "split_strategy": split.strategy,
        "train_period": split.train_period,
        "validation_period": split.validation_period,
        "test_period": split.test_period,
        "feature_set": "intake_time_v1",
        "random_state": RANDOM_STATE,
        "run_timestamp": run_timestamp,
        "train_rows": len(split.train),
        "validation_rows": len(split.validation),
        "test_rows": len(split.test),
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "params": params,
    }


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

    metadata = _base_metadata(
        model_name="catboost",
        task=task,
        split=split,
        feature_columns=feature_columns,
        categorical_features=categorical_features,
        run_timestamp=run_timestamp,
        params=params,
    )
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
        feature_columns = feature_columns_for(split.train)
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
        rows.append({**metadata, **classification_metrics(split.test["classification_target"], predictions, scores)})
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
    """Train CatBoost days-to-outcome regressors."""
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
        feature_columns = feature_columns_for(split.train)
        model, metadata = _fit_and_save(
            model=CatBoostRegressor(**params),
            task="regression",
            split=split,
            feature_columns=feature_columns,
            target_column="regression_target_days",
            models_dir=models_dir,
            run_timestamp=run_timestamp,
            params=params,
        )
        test_x = prepare_catboost_frame(split.test, feature_columns)
        predictions = model.predict(test_x)
        rows.append({**metadata, **regression_metrics(split.test["regression_target_days"], predictions)})
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
) -> AdvancedTrainingOutputs:
    """Train all advanced CatBoost models and save metric outputs."""
    header = pd.read_csv(data_path, nrows=0)
    parse_dates = [col for col in ["intake_datetime", "outcome_datetime"] if col in header.columns]
    df = pd.read_csv(data_path, parse_dates=parse_dates)
    df = limit_rows(df, max_rows)

    metrics_output_dir = Path(metrics_dir)
    model_output_dir = Path(models_dir)
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now(timezone.utc).isoformat()
    classification = pd.DataFrame(
        train_advanced_classification(
            df,
            model_output_dir,
            run_timestamp,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            early_stopping_rounds=early_stopping_rounds,
        )
    )
    regression = pd.DataFrame(
        train_advanced_regression(
            df,
            model_output_dir,
            run_timestamp,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            early_stopping_rounds=early_stopping_rounds,
        )
    )

    classification.to_csv(metrics_output_dir / "advanced_classification_metrics.csv", index=False)
    regression.to_csv(metrics_output_dir / "advanced_regression_metrics.csv", index=False)
    pd.concat([classification, regression], ignore_index=True, sort=False).to_csv(
        metrics_output_dir / "advanced_metrics.csv",
        index=False,
    )
    return AdvancedTrainingOutputs(classification_metrics=classification, regression_metrics=regression)
