"""Baseline model training for AAC adoption thesis experiments."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aac_adoption.config import RANDOM_STATE
from aac_adoption.features.feature_sets import (
    INTAKE_TIME_FEATURES,
    available_intake_features,
    validate_no_leakage,
)
from aac_adoption.interpretation.explain import (
    append_table,
    logistic_regression_coefficients,
    random_forest_feature_importance,
)
from aac_adoption.models.artifacts import save_model_artifact
from aac_adoption.models.evaluate import classification_metrics, regression_metrics
from aac_adoption.models.split import DatasetSplit, make_time_split


FEATURE_COLUMNS = INTAKE_TIME_FEATURES

NUMERIC_FEATURES = [
    "age_days",
    "age_months",
    "age_years",
    "intake_year",
    "intake_month",
    "intake_quarter",
]
CATEGORICAL_FEATURES = [
    "animal_type",
    "intake_type",
    "intake_condition",
    "sex_upon_intake",
    "age_group",
    "primary_breed",
    "simplified_breed_group",
    "primary_color",
    "simplified_color_group",
    "is_black_or_dark",
    "is_named",
    "intake_season",
    "covid_period",
]
ANIMAL_SUBSETS = ["combined", "dogs", "cats"]


@dataclass(frozen=True)
class BaselineTrainingOutputs:
    """Metric tables returned by baseline training."""

    classification_metrics: pd.DataFrame
    regression_metrics: pd.DataFrame


def _available_features(df: pd.DataFrame, columns: list[str]) -> list[str]:
    features = [column for column in columns if column in df.columns]
    validate_no_leakage(features)
    return features


def feature_columns_for(df: pd.DataFrame) -> list[str]:
    """Return available intake-time model features."""
    features = available_intake_features(_available_features(df, FEATURE_COLUMNS))
    validate_no_leakage(features)
    return features


def make_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing from intake-time predictors only."""
    numeric_features = _available_features(df, NUMERIC_FEATURES)
    categorical_features = _available_features(df, CATEGORICAL_FEATURES)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def limit_rows(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    """Use reproducible sample for fast first baseline runs."""
    if not max_rows or len(df) <= max_rows:
        return df
    if "classification_target" in df.columns and df["classification_target"].nunique() == 2:
        return (
            df.groupby("classification_target", group_keys=False)
            .sample(frac=max_rows / len(df), random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
    return df.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)


def _base_metadata(
    *,
    model_name: str,
    task: str,
    split: DatasetSplit,
    feature_set: str,
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
        "feature_set": feature_set,
        "random_state": RANDOM_STATE,
        "run_timestamp": run_timestamp,
        "train_rows": len(split.train),
        "validation_rows": len(split.validation),
        "test_rows": len(split.test),
    }


def _fit_and_save(
    *,
    model,
    model_name: str,
    task: str,
    split: DatasetSplit,
    feature_columns: list[str],
    target_column: str,
    models_dir: Path,
    run_timestamp: str,
):
    preprocessor = make_preprocessor(split.train[feature_columns])
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(split.train[feature_columns], split.train[target_column])

    metadata = _base_metadata(
        model_name=model_name,
        task=task,
        split=split,
        feature_set="intake_time_v1",
        run_timestamp=run_timestamp,
    )
    path = save_model_artifact(
        pipeline=pipeline,
        base_dir=models_dir,
        task=task,
        animal_subset=split.animal_subset,
        model_name=model_name,
        metadata=metadata,
    )
    metadata["artifact_path"] = str(path)
    return pipeline, metadata


def train_classification_baselines(
    df: pd.DataFrame,
    models_dir: Path,
    tables_dir: Path,
    run_timestamp: str,
) -> list[dict]:
    """Train adoption classification baselines for combined/dog/cat subsets."""
    rows: list[dict] = []
    for subset in ANIMAL_SUBSETS:
        split = make_time_split(df, "classification_target", animal_subset=subset)
        feature_columns = feature_columns_for(split.train)
        models = {
            "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
            "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "random_forest": RandomForestClassifier(
                n_estimators=50,
                max_depth=14,
                min_samples_leaf=10,
                random_state=RANDOM_STATE,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
        }

        for model_name, model in models.items():
            pipeline, metadata = _fit_and_save(
                model=model,
                model_name=model_name,
                task="classification",
                split=split,
                feature_columns=feature_columns,
                target_column="classification_target",
                models_dir=models_dir,
                run_timestamp=run_timestamp,
            )
            predictions = pipeline.predict(split.test[feature_columns])
            scores = (
                pipeline.predict_proba(split.test[feature_columns])[:, 1]
                if hasattr(pipeline, "predict_proba")
                else None
            )
            metrics = classification_metrics(split.test["classification_target"], predictions, scores)
            rows.append({**metadata, **metrics})

            if model_name == "logistic_regression":
                append_table(
                    logistic_regression_coefficients(pipeline, metadata),
                    tables_dir / "logistic_regression_coefficients.csv",
                )
            if model_name == "random_forest":
                append_table(
                    random_forest_feature_importance(pipeline, metadata),
                    tables_dir / "random_forest_feature_importance.csv",
                )
    return rows


def train_regression_baselines(
    df: pd.DataFrame,
    models_dir: Path,
    run_timestamp: str,
) -> list[dict]:
    """Train LOS/time-to-outcome regression baselines for combined/dog/cat subsets."""
    rows: list[dict] = []
    for subset in ANIMAL_SUBSETS:
        split = make_time_split(df, "regression_target_days", animal_subset=subset)
        feature_columns = feature_columns_for(split.train)
        models = {
            "dummy_median": DummyRegressor(strategy="median"),
            "ridge": Ridge(),
            "random_forest": RandomForestRegressor(
                n_estimators=50,
                max_depth=14,
                min_samples_leaf=10,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        }

        for model_name, model in models.items():
            pipeline, metadata = _fit_and_save(
                model=model,
                model_name=model_name,
                task="regression",
                split=split,
                feature_columns=feature_columns,
                target_column="regression_target_days",
                models_dir=models_dir,
                run_timestamp=run_timestamp,
            )
            predictions = pipeline.predict(split.test[feature_columns])
            metrics = regression_metrics(split.test["regression_target_days"], predictions)
            rows.append({**metadata, **metrics})
    return rows


def train_all_baselines(
    data_path: str | Path,
    metrics_dir: str | Path = "reports/metrics",
    models_dir: str | Path = "models/baseline",
    tables_dir: str | Path = "reports/tables",
    max_rows: int | None = None,
    output_path: str | Path | None = None,
) -> BaselineTrainingOutputs:
    """Train all baseline models and save metric/interpretability outputs."""
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
        table_output_dir / "logistic_regression_coefficients.csv",
        table_output_dir / "random_forest_feature_importance.csv",
    ]:
        if stale.exists():
            stale.unlink()

    run_timestamp = datetime.now(timezone.utc).isoformat()
    classification_rows = train_classification_baselines(
        df=df,
        models_dir=model_output_dir,
        tables_dir=table_output_dir,
        run_timestamp=run_timestamp,
    )
    regression_rows = train_regression_baselines(
        df=df,
        models_dir=model_output_dir,
        run_timestamp=run_timestamp,
    )

    classification = pd.DataFrame(classification_rows)
    regression = pd.DataFrame(regression_rows)
    classification.to_csv(metrics_output_dir / "classification_metrics.csv", index=False)
    regression.to_csv(metrics_output_dir / "regression_metrics.csv", index=False)

    combined = pd.concat([classification, regression], ignore_index=True, sort=False)
    combined.to_csv(metrics_output_dir / "baseline_metrics.csv", index=False)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)

    return BaselineTrainingOutputs(
        classification_metrics=classification,
        regression_metrics=regression,
    )
