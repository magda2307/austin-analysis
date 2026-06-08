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
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from aac_adoption.config import RANDOM_STATE
from aac_adoption.features.feature_sets import (
    INTAKE_TIME_FEATURES,
    available_features_for_df,
    model_feature_columns,
)
from aac_adoption.interpretation.explain import (
    append_table,
    logistic_regression_coefficients,
    random_forest_feature_importance,
)
from aac_adoption.models.artifacts import save_model_artifact
from aac_adoption.models.evaluate import classification_metrics, regression_metrics
from aac_adoption.models.split import DatasetSplit, make_time_split
from aac_adoption.models.metadata import base_training_metadata


FEATURE_COLUMNS = INTAKE_TIME_FEATURES

NUMERIC_FEATURES = [
    "age_days",
    "intake_year",
    "intake_month",
    "daily_temp_max",
    "daily_temp_min",
    "daily_precipitation",
    "animal_311_requests_7d",
    "animal_311_requests_30d",
    "intake_volume_7d",
    "intake_volume_30d",
]
CATEGORICAL_FEATURES = [
    "animal_type",
    "intake_type",
    "intake_condition",
    "sex_upon_intake",
    "age_group",
    "primary_breed",
    "simplified_breed_group",
    "found_location_kind",
    "found_location_area",
    "is_austin_found_location",
    "is_outside_jurisdiction",
    "is_intersection_location",
    "is_address_like_location",
    "is_airport_location",
    "primary_color",
    "simplified_color_group",
    "is_black_or_dark",
    "is_named",
    "covid_period",
    "is_extreme_heat",
    "is_rainy_day",
]
ANIMAL_SUBSETS = ["combined", "dogs", "cats"]


def categorical_to_object(df: pd.DataFrame) -> pd.DataFrame:
    """Keep categorical preprocessing stable across sklearn/pandas versions."""
    return df.astype("object")


@dataclass(frozen=True)
class BaselineTrainingOutputs:
    """Metric tables returned by baseline training."""

    classification_metrics: pd.DataFrame
    regression_metrics: pd.DataFrame


def make_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing from intake-time predictors only."""
    numeric_features = available_features_for_df(df, NUMERIC_FEATURES)
    categorical_features = available_features_for_df(df, CATEGORICAL_FEATURES)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("as_object", FunctionTransformer(categorical_to_object, feature_names_out="one-to-one")),
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
    dataset_path: str,
    winsorize_target: bool = False,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
):
    from aac_adoption.features.feature_engineering import winsorize_outliers
    from sklearn.base import clone
    
    # Train-only winsorization: compute quantiles on train set and apply only during training
    train_target = split.train[target_column].copy()
    metadata = {
        "training_target_min": float(train_target.min()),
        "training_target_max": float(train_target.max()),
    }
    if winsorize_target and "regression" in task:
        lower = train_target.quantile(lower_quantile)
        upper = train_target.quantile(upper_quantile)
        train_target = winsorize_outliers(train_target, lower_quantile, upper_quantile)
        metadata["winsorization_lower_quantile"] = lower_quantile
        metadata["winsorization_upper_quantile"] = upper_quantile
        metadata["winsorization_lower_value"] = lower
        metadata["winsorization_upper_value"] = upper
    
    preprocessor = make_preprocessor(split.train[feature_columns])
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", clone(model))])
    
    fit_params = {}
    if "sample_weight" in split.train.columns and "dummy" not in model_name:
        fit_params["model__sample_weight"] = split.train["sample_weight"]

    pipeline.fit(split.train[feature_columns], train_target, **fit_params)

    metadata.update(base_training_metadata(
        model_name=model_name,
        task=task,
        split=split,
        feature_columns=feature_columns,
        run_timestamp=run_timestamp,
        target_column=target_column,
        dataset_path=dataset_path,
    ))
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
    dataset_path: str,
) -> list[dict]:
    """Train adoption classification baselines for combined/dog/cat subsets."""
    rows: list[dict] = []
    for subset in ANIMAL_SUBSETS:
        split = make_time_split(df, "classification_target", animal_subset=subset)
        feature_columns = model_feature_columns(split.train)
        models = {
            "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
            "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
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
                dataset_path=dataset_path,
            )
            if not split.selection.empty:
                sel_predictions = pipeline.predict(split.selection[feature_columns])
                sel_scores = (
                    pipeline.predict_proba(split.selection[feature_columns])[:, 1]
                    if hasattr(pipeline, "predict_proba") and pipeline.predict_proba(split.selection[feature_columns]).shape[1] == 2
                    else None
                )
                sel_metrics = classification_metrics(
                    split.selection["classification_target"], 
                    sel_predictions, 
                    sel_scores,
                    compute_ci=(subset in ["dogs", "cats"]),
                )
                rows.append({
                    **metadata,
                    **sel_metrics,
                    "target_column": "classification_target",
                    "target_transform": "identity",
                    "prediction_inverse_transform": "identity",
                    "metric_split": "selection",
                    "selection_eligible": 1,
                })

            if not split.test.empty:
                predictions = pipeline.predict(split.test[feature_columns])
                scores = (
                    pipeline.predict_proba(split.test[feature_columns])[:, 1]
                    if hasattr(pipeline, "predict_proba") and pipeline.predict_proba(split.test[feature_columns]).shape[1] == 2
                    else None
                )
                metrics = classification_metrics(
                    split.test["classification_target"], 
                    predictions, 
                    scores,
                    compute_ci=(subset in ["dogs", "cats"]),
                )
                rows.append({
                    **metadata,
                    **metrics,
                    "target_column": "classification_target",
                    "target_transform": "identity",
                    "prediction_inverse_transform": "identity",
                    "metric_split": "test",
                    "selection_eligible": 0,
                })

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
    dataset_path: str,
) -> list[dict]:
    """Train LOS/time-to-outcome regression baselines for combined/dog/cat subsets."""
    rows: list[dict] = []
    for subset in ANIMAL_SUBSETS:
        split = make_time_split(df, "regression_target_days", animal_subset=subset)
        feature_columns = model_feature_columns(split.train)
        models = {
            "dummy_median": DummyRegressor(strategy="median"),
            "ridge": Ridge(),
            "random_forest": RandomForestRegressor(
                n_estimators=200,
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
                dataset_path=dataset_path,
                winsorize_target=True,
            )
            if not split.selection.empty:
                sel_predictions = pipeline.predict(split.selection[feature_columns])
                sel_metrics = regression_metrics(
                    split.selection["regression_target_days"], 
                    sel_predictions,
                    compute_ci=(subset in ["dogs", "cats"]),
                )
                rows.append({
                    **metadata,
                    **sel_metrics,
                    "target_column": "regression_target_days",
                    "target_transform": "identity",
                    "prediction_inverse_transform": "identity",
                    "metric_split": "selection",
                    "selection_eligible": 1,
                })

            if not split.test.empty:
                predictions = pipeline.predict(split.test[feature_columns])
                metrics = regression_metrics(
                    split.test["regression_target_days"], 
                    predictions,
                    compute_ci=(subset in ["dogs", "cats"]),
                )
                rows.append({
                    **metadata,
                    **metrics,
                    "target_column": "regression_target_days",
                    "target_transform": "identity",
                    "prediction_inverse_transform": "identity",
                    "metric_split": "test",
                    "selection_eligible": 0,
                })
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
        dataset_path=str(data_path),
    )
    regression_rows = train_regression_baselines(
        df=df,
        models_dir=model_output_dir,
        run_timestamp=run_timestamp,
        dataset_path=str(data_path),
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
