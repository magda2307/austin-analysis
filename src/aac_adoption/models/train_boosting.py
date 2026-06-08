"""Gradient boosting training for AAC adoption thesis experiments."""

import json
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
    available_features_for_df,
    model_feature_columns,
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
from aac_adoption.models.metadata import base_training_metadata


@dataclass(frozen=True)
class BoostingTrainingOutputs:
    """Metric tables returned by boosting training."""

    classification_metrics: pd.DataFrame
    regression_metrics: pd.DataFrame


def make_boosting_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Create dense preprocessing compatible with HistGradientBoosting."""
    numeric_features = available_features_for_df(df, NUMERIC_FEATURES)
    categorical_features = available_features_for_df(df, CATEGORICAL_FEATURES)
    
    native_categorical = [
        "intake_type",
        "intake_condition", 
        "sex_upon_intake",
        "age_group",
        "simplified_breed_group",
        "simplified_color_group",
        "found_location_kind",
        "intake_season",
        "covid_period",
    ]
    categorical_features = [c for c in categorical_features if c in native_categorical]
    
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


def _fit_and_save(
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
    
    # Train-only winsorization for regression tasks
    train_y = split.train[target_column].copy()
    metadata = {
        "training_target_min": float(train_y.min()),
        "training_target_max": float(train_y.max()),
    }
    if winsorize_target and "regression" in task:
        train_y = winsorize_outliers(train_y, lower_quantile, upper_quantile)
        metadata["winsorization_lower_quantile"] = lower_quantile
        metadata["winsorization_upper_quantile"] = upper_quantile
        metadata["winsorization_lower_value"] = float(train_y.quantile(lower_quantile))
        metadata["winsorization_upper_value"] = float(train_y.quantile(upper_quantile))
    
    pipeline = Pipeline(
        steps=[
            ("preprocess", make_boosting_preprocessor(split.train[feature_columns])),
            ("model", clone(model)),
        ]
    )
    
    fit_params = {}
    if "sample_weight" in split.train.columns:
        fit_params["model__sample_weight"] = split.train["sample_weight"]
        
    pipeline.fit(split.train[feature_columns], train_y, **fit_params)
    metadata.update(base_training_metadata(
        model_name=model_name,
        task=task,
        split=split,
        feature_columns=feature_columns,
        run_timestamp=run_timestamp,
        target_column=target_column,
        dataset_path=dataset_path,
    ))
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
    if split.validation.empty:
        raise ValueError(
            f"Permutation importance requires validation data for {split.animal_subset}. "
            f"Validation split is empty. This violates validation-only methodology. "
            f"Check time-based split parameters (validation_years=2022-2023)."
        )
    sample = split.validation
    importance_split = "validation"
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
    table = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "importance_split": importance_split,
            "evaluation_period": importance_split,
        }
    )
    for key, val in metadata.items():
        table[key] = json.dumps(val) if isinstance(val, (list, dict)) else val
    return table.sort_values("importance_mean", ascending=False)


def train_boosting_classification(
    df: pd.DataFrame,
    models_dir: Path,
    tables_dir: Path,
    run_timestamp: str,
    dataset_path: str,
    permutation_repeats: int,
    permutation_max_rows: int,
    **kwargs,
) -> list[dict]:
    rows: list[dict] = []
    
    model_params = {
        "learning_rate": 0.08,
        "max_iter": 100,
        "max_leaf_nodes": 31,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
    }
    model_params.update(kwargs)

    for subset in ANIMAL_SUBSETS:
        split = make_time_split(df, "classification_target", animal_subset=subset)
        feature_columns = model_feature_columns(split.train)
        pipeline, metadata = _fit_and_save(
            model=HistGradientBoostingClassifier(**model_params),
            model_name="hist_gradient_boosting",
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
            sel_scores = pipeline.predict_proba(split.selection[feature_columns])[:, 1]
            sel_metrics = classification_metrics(
                split.selection["classification_target"], 
                sel_predictions, 
                sel_scores,
                compute_ci=(subset in ["dogs", "cats"])
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
            test_predictions = pipeline.predict(split.test[feature_columns])
            test_scores = pipeline.predict_proba(split.test[feature_columns])[:, 1]
            test_metrics = classification_metrics(
                split.test["classification_target"], 
                test_predictions, 
                test_scores,
                compute_ci=(subset in ["dogs", "cats"])
            )
            rows.append({
                **metadata,
                **test_metrics,
                "target_column": "classification_target",
                "target_transform": "identity",
                "prediction_inverse_transform": "identity",
                "metric_split": "test",
                "selection_eligible": 0,
            })
        append_table(
            _permutation_table(
                pipeline,
                split,
                feature_columns,
                "classification_target",
                metadata,
                scoring="average_precision",
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
    dataset_path: str,
    permutation_repeats: int,
    permutation_max_rows: int,
    **kwargs,
) -> list[dict]:
    rows: list[dict] = []

    model_params = {
        "learning_rate": 0.08,
        "max_iter": 100,
        "max_leaf_nodes": 31,
        "random_state": RANDOM_STATE,
    }
    model_params.update(kwargs)

    for subset in ANIMAL_SUBSETS:
        split = make_time_split(df, "regression_target_days", animal_subset=subset)
        feature_columns = model_feature_columns(split.train)
        pipeline, metadata = _fit_and_save(
            model=HistGradientBoostingRegressor(**model_params),
            model_name="hist_gradient_boosting",
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
                compute_ci=(subset in ["dogs", "cats"])
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
            test_predictions = pipeline.predict(split.test[feature_columns])
            test_metrics = regression_metrics(
                split.test["regression_target_days"], 
                test_predictions,
                compute_ci=(subset in ["dogs", "cats"])
            )
            rows.append({
                **metadata,
                **test_metrics,
                "target_column": "regression_target_days",
                "target_transform": "identity",
                "prediction_inverse_transform": "identity",
                "metric_split": "test",
                "selection_eligible": 0,
            })
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
    tuned_params_path: str | Path | None = None,
    allow_default_params: bool = False,
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

    clf_kwargs = {}
    reg_kwargs = {}
    
    if tuned_params_path is not None:
        import json
        p = Path(tuned_params_path)
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            for key, kwargs_dict in [("hist_gradient_boosting_classification", clf_kwargs), ("hist_gradient_boosting_regression", reg_kwargs)]:
                if key in d:
                    tune_info = d[key]
                    if tune_info.get("status") == "failed":
                        if not allow_default_params:
                            raise ValueError(f"Tuning failed for {key}. Explicit development flag required to use defaults.")
                    elif tune_info.get("best_params") is not None:
                        kwargs_dict.update(tune_info["best_params"])

    run_timestamp = datetime.now(timezone.utc).isoformat()
    classification = pd.DataFrame(
        train_boosting_classification(
            df,
            model_output_dir,
            table_output_dir,
            run_timestamp,
            str(data_path),
            permutation_repeats,
            permutation_max_rows,
            **clf_kwargs
        )
    )
    regression = pd.DataFrame(
        train_boosting_regression(
            df,
            model_output_dir,
            table_output_dir,
            run_timestamp,
            str(data_path),
            permutation_repeats,
            permutation_max_rows,
            **reg_kwargs
        )
    )

    classification.to_csv(metrics_output_dir / "boosting_classification_metrics.csv", index=False)
    regression.to_csv(metrics_output_dir / "boosting_regression_metrics.csv", index=False)
    pd.concat([classification, regression], ignore_index=True, sort=False).to_csv(
        metrics_output_dir / "boosting_metrics.csv",
        index=False,
    )
    return BoostingTrainingOutputs(classification_metrics=classification, regression_metrics=regression)
