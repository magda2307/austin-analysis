"""Gradient boosting training for AAC adoption thesis experiments."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from aac_adoption.config import RANDOM_STATE
from aac_adoption.features.feature_sets import (
    INTAKE_TIME_FEATURES,
    available_intake_features,
    feature_set_label,
    validate_no_leakage,
)
from aac_adoption.interpretation.explain import append_table
from aac_adoption.models.artifacts import save_model_artifact
from aac_adoption.models.evaluate import classification_metrics, regression_metrics
from aac_adoption.models.split import DatasetSplit, make_time_split
from aac_adoption.models.train_baseline import (
    ANIMAL_SUBSETS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    categorical_to_object,
    limit_rows,
)


@dataclass(frozen=True)
class BoostingTrainingOutputs:
    """Metric tables returned by boosting training."""

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


def make_boosting_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Create dense preprocessing compatible with HistGradientBoosting."""
    numeric_features = _available_features(df, NUMERIC_FEATURES)
    categorical_features = _available_features(df, CATEGORICAL_FEATURES)
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("as_object", FunctionTransformer(categorical_to_object, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        sparse_threshold=0.0,
    )


def _base_metadata(
    model_name: str,
    task: str,
    split: DatasetSplit,
    feature_columns: list[str],
    run_timestamp: str,
) -> dict:
    return {
        "model_name": model_name,
        "task": task,
        "animal_subset": split.animal_subset,
        "split_strategy": split.strategy,
        "train_period": split.train_period,
        "validation_period": split.validation_period,
        "test_period": split.test_period,
        "feature_set": feature_set_label(feature_columns),
        "random_state": RANDOM_STATE,
        "run_timestamp": run_timestamp,
        "train_rows": len(split.train),
        "validation_rows": len(split.validation),
        "test_rows": len(split.test),
    }


def _fit_and_save(
    model,
    model_name: str,
    task: str,
    split: DatasetSplit,
    feature_columns: list[str],
    target_column: str,
    models_dir: Path,
    run_timestamp: str,
):
    pipeline = Pipeline(
        steps=[
            ("preprocess", make_boosting_preprocessor(split.train[feature_columns])),
            ("model", model),
        ]
    )
    pipeline.fit(split.train[feature_columns], split.train[target_column])
    metadata = _base_metadata(model_name, task, split, feature_columns, run_timestamp)
    path = save_model_artifact(pipeline, models_dir, task, split.animal_subset, model_name, metadata)
    metadata["artifact_path"] = str(path)
    return pipeline, metadata


def _permutation_table(
    pipeline,
    split: DatasetSplit,
    feature_columns: list[str],
    target_column: str,
    metadata: dict,
    scoring: str,
    repeats: int,
    max_rows: int,
) -> pd.DataFrame:
    sample = split.test
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=RANDOM_STATE)
    result = permutation_importance(
        pipeline,
        sample[feature_columns],
        sample[target_column],
        n_repeats=repeats,
        random_state=RANDOM_STATE,
        scoring=scoring,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            **metadata,
        }
    ).sort_values("importance_mean", ascending=False)


def train_boosting_classification(
    df: pd.DataFrame,
    models_dir: Path,
    tables_dir: Path,
    run_timestamp: str,
    permutation_repeats: int,
    permutation_max_rows: int,
) -> list[dict]:
    rows: list[dict] = []
    for subset in ANIMAL_SUBSETS:
        split = make_time_split(df, "classification_target", animal_subset=subset)
        feature_columns = feature_columns_for(split.train)
        pipeline, metadata = _fit_and_save(
            model=HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=100,
                max_leaf_nodes=31,
                random_state=RANDOM_STATE,
            ),
            model_name="hist_gradient_boosting",
            task="classification",
            split=split,
            feature_columns=feature_columns,
            target_column="classification_target",
            models_dir=models_dir,
            run_timestamp=run_timestamp,
        )
        predictions = pipeline.predict(split.test[feature_columns])
        scores = pipeline.predict_proba(split.test[feature_columns])[:, 1]
        rows.append({**metadata, **classification_metrics(split.test["classification_target"], predictions, scores)})
        append_table(
            _permutation_table(
                pipeline,
                split,
                feature_columns,
                "classification_target",
                metadata,
                scoring="roc_auc",
                repeats=permutation_repeats,
                max_rows=permutation_max_rows,
            ),
            tables_dir / "permutation_importance_classification.csv",
        )
    return rows


def train_boosting_regression(
    df: pd.DataFrame,
    models_dir: Path,
    tables_dir: Path,
    run_timestamp: str,
    permutation_repeats: int,
    permutation_max_rows: int,
) -> list[dict]:
    rows: list[dict] = []
    for subset in ANIMAL_SUBSETS:
        split = make_time_split(df, "regression_target_days", animal_subset=subset)
        feature_columns = feature_columns_for(split.train)
        pipeline, metadata = _fit_and_save(
            model=HistGradientBoostingRegressor(
                learning_rate=0.08,
                max_iter=100,
                max_leaf_nodes=31,
                random_state=RANDOM_STATE,
            ),
            model_name="hist_gradient_boosting",
            task="regression",
            split=split,
            feature_columns=feature_columns,
            target_column="regression_target_days",
            models_dir=models_dir,
            run_timestamp=run_timestamp,
        )
        predictions = pipeline.predict(split.test[feature_columns])
        rows.append({**metadata, **regression_metrics(split.test["regression_target_days"], predictions)})
        append_table(
            _permutation_table(
                pipeline,
                split,
                feature_columns,
                "regression_target_days",
                metadata,
                scoring="neg_mean_absolute_error",
                repeats=permutation_repeats,
                max_rows=permutation_max_rows,
            ),
            tables_dir / "permutation_importance_regression.csv",
        )
    return rows


def train_all_boosting(
    data_path: str | Path,
    metrics_dir: str | Path = "reports/metrics",
    models_dir: str | Path = "models/boosting",
    tables_dir: str | Path = "reports/tables",
    max_rows: int | None = None,
    permutation_repeats: int = 3,
    permutation_max_rows: int = 3000,
) -> BoostingTrainingOutputs:
    """Train all sklearn gradient boosting models and save outputs."""
    header = pd.read_csv(data_path, nrows=0)
    parse_dates = [col for col in ["intake_datetime", "outcome_datetime"] if col in header.columns]
    df = pd.read_csv(data_path, parse_dates=parse_dates)
    df = limit_rows(df, max_rows)

    metrics_output_dir = Path(metrics_dir)
    model_output_dir = Path(models_dir)
    table_output_dir = Path(tables_dir)
    metrics_output_dir.mkdir(parents=True, exist_ok=True)
    table_output_dir.mkdir(parents=True, exist_ok=True)
    for stale in [
        table_output_dir / "permutation_importance_classification.csv",
        table_output_dir / "permutation_importance_regression.csv",
    ]:
        if stale.exists():
            stale.unlink()

    run_timestamp = datetime.now(timezone.utc).isoformat()
    classification = pd.DataFrame(
        train_boosting_classification(
            df,
            model_output_dir,
            table_output_dir,
            run_timestamp,
            permutation_repeats,
            permutation_max_rows,
        )
    )
    regression = pd.DataFrame(
        train_boosting_regression(
            df,
            model_output_dir,
            table_output_dir,
            run_timestamp,
            permutation_repeats,
            permutation_max_rows,
        )
    )

    classification.to_csv(metrics_output_dir / "boosting_classification_metrics.csv", index=False)
    regression.to_csv(metrics_output_dir / "boosting_regression_metrics.csv", index=False)
    pd.concat([classification, regression], ignore_index=True, sort=False).to_csv(
        metrics_output_dir / "boosting_metrics.csv",
        index=False,
    )
    return BoostingTrainingOutputs(classification_metrics=classification, regression_metrics=regression)
