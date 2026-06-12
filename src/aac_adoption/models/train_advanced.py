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
from aac_adoption.models.artifacts import artifact_path, save_model_artifact
from aac_adoption.models.evaluate import classification_metrics, regression_metrics
from aac_adoption.models.split import DatasetSplit, make_time_split
from aac_adoption.models.train_baseline import ANIMAL_SUBSETS, CATEGORICAL_FEATURES, limit_rows
from aac_adoption.models.metadata import base_training_metadata
from aac_adoption.analysis.survival_analysis import log_transform_LOS


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
    dataset_path: str,
    params: dict[str, Any],
    winsorize_target: bool = False,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> tuple[Any, dict[str, Any]]:
    from aac_adoption.features.feature_engineering import winsorize_outliers
    
    categorical_features = categorical_features_for(feature_columns)
    train_x = prepare_catboost_frame(split.train, feature_columns)
    eval_df = split.calibration if (split.calibration is not None and not split.calibration.empty) else split.validation
    validation_x = prepare_catboost_frame(eval_df, feature_columns) if (eval_df is not None and not eval_df.empty) else None
    
    # Train-only winsorization for regression tasks
    train_y = split.train[target_column].copy()
    metadata = {'training_target_min': float(train_y.min()), 'training_target_max': float(train_y.max())}
    if winsorize_target and "regression" in task:
        train_y = winsorize_outliers(train_y, lower_quantile, upper_quantile)
        metadata["winsorization_lower_quantile"] = lower_quantile
        metadata["winsorization_upper_quantile"] = upper_quantile
        metadata["winsorization_lower_value"] = float(train_y.quantile(lower_quantile))
        metadata["winsorization_upper_value"] = float(train_y.quantile(upper_quantile))
    
    fit_kwargs: dict[str, Any] = {
        "X": train_x,
        "y": train_y,
        "cat_features": categorical_features,
        "verbose": False,
    }
    if "sample_weight" in split.train.columns:
        fit_kwargs["sample_weight"] = split.train["sample_weight"]
    if validation_x is not None:
        fit_kwargs["eval_set"] = (validation_x, eval_df[target_column])
        fit_kwargs["use_best_model"] = True
    model.fit(**fit_kwargs)

    metadata.update(base_training_metadata(
        model_name="catboost",
        task=task,
        split=split,
        feature_columns=feature_columns,
        run_timestamp=run_timestamp,
        target_column=target_column,
        dataset_path=dataset_path,
        categorical_features=categorical_features,
        params=params,
    ))
    path = save_model_artifact(model, models_dir, task, split.animal_subset, "catboost", metadata)
    metadata["artifact_path"] = str(path)
    return model, metadata


from aac_adoption.models.calibrate import apply_calibration_to_predictions


def _split_validation_for_calibration(validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split validation chronologically into early-stop and calibration frames."""
    if validation.empty or len(validation) < 4:
        return validation, validation.iloc[0:0].copy()
    sorted_validation = validation.sort_values("intake_datetime") if "intake_datetime" in validation.columns else validation
    midpoint = len(sorted_validation) // 2
    return sorted_validation.iloc[:midpoint].copy(), sorted_validation.iloc[midpoint:].copy()

def train_advanced_classification(
    df: pd.DataFrame,
    models_dir: Path,
    run_timestamp: str,
    dataset_path: str,
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    early_stopping_rounds: int,
) -> list[dict[str, Any]]:
    """Train CatBoost adoption classifiers with post-hoc calibration."""
    rows: list[dict[str, Any]] = []
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "early_stopping_rounds": early_stopping_rounds,
        "random_seed": RANDOM_STATE,
        "auto_class_weights": "Balanced",
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
            dataset_path=dataset_path,
            params=params,
        )
        test_x = prepare_catboost_frame(split.test, feature_columns)
        
        if not split.selection.empty:
            sel_x = prepare_catboost_frame(split.selection, feature_columns)
            sel_predictions = model.predict(sel_x).astype(int)
            sel_scores = model.predict_proba(sel_x)[:, 1]
            sel_metrics = classification_metrics(
                split.selection["classification_target"], 
                sel_predictions, 
                sel_scores,
                compute_ci=(subset in ["dogs", "cats"])
            )
            rows.append({
                **metadata,
                **sel_metrics,
                "calibration_method": None,
                "target_column": "classification_target",
                "target_transform": "identity",
                "prediction_inverse_transform": "identity",
                "metric_split": "selection",
                "selection_eligible": 1,
            })
            
        if not split.test.empty:
            test_predictions = model.predict(test_x).astype(int)
            test_scores = model.predict_proba(test_x)[:, 1]
            test_metrics = classification_metrics(
                split.test["classification_target"], 
                test_predictions, 
                test_scores,
                compute_ci=(subset in ["dogs", "cats"])
            )
            rows.append({
                **metadata,
                **test_metrics,
                "calibration_method": None,
                "target_column": "classification_target",
                "target_transform": "identity",
                "prediction_inverse_transform": "identity",
                "metric_split": "test",
                "selection_eligible": 0,
            })
        
        # Apply Post-Hoc Calibration if calibration set is available
        if split.calibration is not None and not split.calibration.empty:
            val_x = prepare_catboost_frame(split.calibration, feature_columns)
            calibrated_model = apply_calibration_to_predictions(
                base_model=model,
                X_train=prepare_catboost_frame(split.train, feature_columns),
                y_train=split.train["classification_target"],
                X_calib=val_x,
                y_calib=split.calibration["classification_target"],
                calib_method="isotonic"
            )
            
            # Save calibrated artifact
            calibrated_metadata = {
                **metadata,
                "model_name": "catboost_calibrated",
                "task": "classification_calibrated",
                "base_model_name": "catboost",
                "base_artifact_path": metadata["artifact_path"],
                "calibration_method": "isotonic",
                "calibration_rows": len(split.calibration),
                "early_stopping_rows": len(split.calibration) if (split.calibration is not None and not split.calibration.empty) else len(split.validation),
            }
            calibrated_metadata["artifact_path"] = str(
                artifact_path(
                    models_dir,
                    "classification_calibrated",
                    split.animal_subset,
                    "catboost_calibrated",
                )
            )
            calibrated_path = save_model_artifact(
                calibrated_model,
                models_dir,
                "classification_calibrated",
                split.animal_subset,
                "catboost_calibrated",
                calibrated_metadata,
            )
            
            if not split.selection.empty:
                sel_x = prepare_catboost_frame(split.selection, feature_columns)
                sel_calib_preds = calibrated_model.predict(sel_x).astype(int)
                sel_calib_scores = calibrated_model.predict_proba(sel_x)[:, 1]
                sel_calib_metrics = classification_metrics(
                    split.selection["classification_target"], 
                    sel_calib_preds, 
                    sel_calib_scores,
                    compute_ci=False
                )
                rows.append({
                    **calibrated_metadata,
                    **sel_calib_metrics,
                    "target_column": "classification_target",
                    "target_transform": "identity",
                    "prediction_inverse_transform": "identity",
                    "metric_split": "selection",
                    "selection_eligible": 1,
                })

            if not split.test.empty:
                calib_predictions = calibrated_model.predict(test_x).astype(int)
                calib_scores = calibrated_model.predict_proba(test_x)[:, 1]
                calib_metrics = classification_metrics(
                    split.test["classification_target"], 
                    calib_predictions, 
                    calib_scores,
                    compute_ci=False
                )
                rows.append({
                    **calibrated_metadata,
                    **calib_metrics,
                    "target_column": "classification_target",
                    "target_transform": "identity",
                    "prediction_inverse_transform": "identity",
                    "metric_split": "test",
                    "selection_eligible": 0,
                })
    return rows


def train_advanced_regression(
    df: pd.DataFrame,
    models_dir: Path,
    run_timestamp: str,
    dataset_path: str,
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    early_stopping_rounds: int,
) -> list[dict[str, Any]]:
    """Train CatBoost days-to-outcome regressors with log-transform."""
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
        
        split_train = split.train.copy()
        split_train = log_transform_LOS(split_train, "regression_target_days")
        
        split_calib = split.calibration.copy() if split.calibration is not None else None
        if split_calib is not None:
            split_calib = log_transform_LOS(split_calib, "regression_target_days")
            
        split_val = split.validation.copy()
        split_val = log_transform_LOS(split_val, "regression_target_days")
        
        split_test = split.test.copy()
        split_test = log_transform_LOS(split_test, "regression_target_days")
        
        model, metadata = _fit_and_save(
            model=CatBoostRegressor(**params),
            task="regression",
            split=DatasetSplit(
                full_data=split.full_data,
                train=split_train,
                calibration=split_calib,
                validation=split_val,
                test=split_test,
                strategy=split.strategy,
                train_period=split.train_period,
                validation_period=split.validation_period,
                test_period=split.test_period,
                animal_subset=split.animal_subset,
                selection=split.selection,
            ),
            feature_columns=feature_columns,
            target_column="log_regression_target_days",
            models_dir=models_dir,
            run_timestamp=run_timestamp,
            dataset_path=dataset_path,
            params=params,
            winsorize_target=True,
        )
        if not split.selection.empty:
            sel_x = prepare_catboost_frame(split.selection, feature_columns)
            sel_predictions = np.exp(model.predict(sel_x)) - 1
            sel_metrics = regression_metrics(split.selection["regression_target_days"], sel_predictions)
            rows.append({
                **metadata,
                **sel_metrics,
                "target_column": "regression_target_days",
                "target_transform": "log1p",
                "prediction_inverse_transform": "expm1",
                "metric_split": "selection",
                "selection_eligible": 1,
            })
            
        if not split.test.empty:
            test_x = prepare_catboost_frame(split.test, feature_columns)
            predictions = np.exp(model.predict(test_x)) - 1
            metrics = regression_metrics(split.test["regression_target_days"], predictions)
            rows.append({
                **metadata,
                **metrics,
                "target_column": "regression_target_days",
                "target_transform": "log1p",
                "prediction_inverse_transform": "expm1",
                "metric_split": "test",
                "selection_eligible": 0,
            })
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
    allow_default_params: bool = False,
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
            for key, kwargs_dict in [("catboost_classification", clf_kwargs), ("catboost_regression", reg_kwargs)]:
                if key in d:
                    tune_info = d[key]
                    best_params = tune_info.get("best_params")
                    if tune_info.get("status") == "failed" or not isinstance(best_params, dict):
                        if not allow_default_params:
                            raise ValueError(f"Tuning failed or missing/malformed parameters for {key}. Explicit development flag required to use defaults.")
                    else:
                        kwargs_dict.update(best_params)

    run_timestamp = datetime.now(timezone.utc).isoformat()
    classification = pd.DataFrame(
        train_advanced_classification(
            df,
            model_output_dir,
            run_timestamp,
            str(data_path),
            early_stopping_rounds=early_stopping_rounds,
            **clf_kwargs
        )
    )
    regression = pd.DataFrame(
        train_advanced_regression(
            df,
            model_output_dir,
            run_timestamp,
            str(data_path),
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
