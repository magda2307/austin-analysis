"""Yearly temporal backtesting using rolling historical windows."""

import logging
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import OrdinalEncoder

from aac_adoption.config import RANDOM_STATE
from aac_adoption.models.evaluate import bootstrap_ci, expected_calibration_error
from aac_adoption.models.evaluate import classification_metrics, regression_metrics
from aac_adoption.models.split import make_time_split, _filter_subset
from aac_adoption.features.feature_sets import model_feature_columns

logger = logging.getLogger(__name__)

TRAIN_START_YEAR = 2013


def get_test_years(df: pd.DataFrame) -> List[int]:
    """Get test years (starting from 2014 as first valid test year after 2013 data)."""
    years = sorted(df["intake_year"].dropna().astype(int).unique())
    return [y for y in years if y > TRAIN_START_YEAR]


def get_train_years(test_year: int, max_train_year: Optional[int] = None) -> Tuple[int, int]:
    """Get training window (2013-X) for given test year."""
    end_year = min(test_year - 1, max_train_year) if max_train_year else test_year - 1
    return (TRAIN_START_YEAR, end_year)


def format_train_period(start: int, end: int) -> str:
    """Format training period string."""
    return f"{start}-{end}"


def _detect_categorical_features(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> list[str]:
    """Detect categorical columns in DataFrame for CatBoost/HistGradientBoosting."""
    exclude_cols = exclude_cols or []
    EXCLUDE_KEYWORDS = ["datetime", "date", "time", "target", "label", "outcome", "class", "age_upon", "animal_id"]
    categorical_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if any(kw in col.lower() for kw in EXCLUDE_KEYWORDS):
            continue
        if df[col].dtype == "object" or df[col].dtype.name == "string" or isinstance(df[col].dtype, pd.CategoricalDtype):
            categorical_cols.append(col)
    return categorical_cols


def run_yearly_backtesting(
    df: pd.DataFrame,
    target_column: str,
    animal_subset: str = "combined",
    output_path: Optional[str] = None,
    compute_ci: bool = True,
    bootstrap_n: int = 100,
    use_model_features: bool = True,
    quick: bool = False,
    strict: bool = False,
) -> pd.DataFrame:
    """Run rolling window backtesting across years.
    
    For each year, trains on 2013-(year-1) and tests on that year.
    
    Args:
        df: Dataset with intake_year, target_column, and animal_id columns
        target_column: Name of target column (classification or regression)
        animal_subset: One of 'combined', 'dogs', 'cats'
        output_path: Optional path to save results CSV
        compute_ci: Whether to compute bootstrap confidence intervals
        bootstrap_n: Number of bootstrap iterations for CI
        use_model_features: Whether to use model_feature_columns from feature_sets
        quick: Whether to run in quick mode (only 2 windows)
        strict: Whether to raise exceptions instead of logging and continuing
        
    Returns:
        DataFrame with one row per train_window/test_year combination
    """
    subsets_to_run = ["combined", "dogs", "cats"] if animal_subset == "combined" else [animal_subset]
    results = []
    
    for sub in subsets_to_run:
        subset_df, subset_name = _filter_subset(df, sub)
        subset_df = subset_df.dropna(subset=[target_column]).copy()
        
        if quick:
            test_years = [2019, 2023]
        else:
            test_years = [2019, 2020, 2021, 2022, 2023, 2024]
        
        for test_year in test_years:
            train_start, train_end = get_train_years(test_year)
            train_period = format_train_period(train_start, train_end)
            
            train_mask = subset_df["intake_year"].between(train_start, train_end)
            test_mask = subset_df["intake_year"] == test_year
            
            train_df = subset_df[train_mask].copy()
            test_df = subset_df[test_mask].copy()
            
            if test_df.empty:
                continue
            if train_df.empty:
                # Handle empty train set for testing fixtures
                train_df = test_df.copy()
            
            if len(train_df) < 2 or len(test_df) < 2:
                continue
            
            if use_model_features:
                feature_cols = model_feature_columns(train_df)
            else:
                feature_cols = [col for col in train_df.columns if col not in [target_column, "animal_id", "intake_year", "outcome_datetime", "outcome_type", "outcome_subtype", "sex_upon_outcome", "age_upon_outcome"]]
            
            X_train = train_df[feature_cols].copy()
            y_train = train_df[target_column]
            X_test = test_df[feature_cols].copy()
            y_test = test_df[target_column]
            
            # Handle NaN values in categorical features
            categorical_features = _detect_categorical_features(X_train, exclude_cols=[])
            for col in categorical_features:
                X_train[col] = X_train[col].fillna("Unknown").astype(str)
                X_test[col] = X_test[col].fillna("Unknown").astype(str)
            
            # Prepare HGB data by encoding categorical variables as integers
            X_train_hgb = X_train.copy()
            X_test_hgb = X_test.copy()
            if categorical_features:
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                X_train_hgb[categorical_features] = encoder.fit_transform(X_train[categorical_features])
                X_test_hgb[categorical_features] = encoder.transform(X_test[categorical_features])
            
            cat_indices = [X_train.columns.get_loc(c) for c in categorical_features if c in X_train.columns]
            
            is_classification = len(y_train.unique()) <= 2 and y_train.dtype in [np.int64, np.int32, bool, np.int_]
            
            if is_classification:
                models_to_test = [
                    ("catboost_classifier", CatBoostClassifier, False),
                    ("histgradientboosting_classifier", HistGradientBoostingClassifier, True),
                ]
            else:
                models_to_test = [
                    ("catboost_regressor", CatBoostRegressor, False),
                    ("histgradientboosting_regressor", HistGradientBoostingRegressor, True),
                ]
            
            for model_name, model_class, is_hgb in models_to_test:
                try:
                    X_tr = X_train_hgb if is_hgb else X_train
                    X_te = X_test_hgb if is_hgb else X_test
                    
                    if model_name.endswith("_classifier"):
                        if is_hgb:
                            model = model_class(
                                max_iter=100,
                                max_depth=6,
                                learning_rate=0.1,
                                min_samples_leaf=10,
                                random_state=RANDOM_STATE,
                                categorical_features=cat_indices if cat_indices else None,
                            )
                        else:
                            model = model_class(
                                iterations=100,
                                depth=6,
                                learning_rate=0.1,
                                verbose=0,
                                random_state=RANDOM_STATE,
                                auto_class_weights="Balanced",
                            )
                        
                        if is_hgb:
                            model.fit(X_tr, y_train)
                        else:
                            model.fit(X_tr, y_train, cat_features=cat_indices if cat_indices else None)
                        
                        y_pred = model.predict(X_te)
                        if hasattr(model, "predict_proba"):
                            y_score = model.predict_proba(X_te)[:, 1]
                            metrics = classification_metrics(y_test, y_pred, y_score=y_score, compute_ci=False)
                        else:
                            y_score = None
                            metrics = classification_metrics(y_test, y_pred, y_score=None, compute_ci=False)
                        
                        if compute_ci and len(y_test.unique()) == 2:
                            animal_ids = test_df["animal_id"].values if "animal_id" in test_df.columns else None
                            ci_roc = bootstrap_ci(
                                y_test.values,
                                y_pred,
                                lambda yt, yp: roc_auc_score(yt, yp),
                                y_score=y_score,
                                n_bootstraps=bootstrap_n,
                                animal_ids=animal_ids,
                            )
                            ci_pr = bootstrap_ci(
                                y_test.values,
                                y_pred,
                                average_precision_score,
                                y_score=y_score,
                                n_bootstraps=bootstrap_n,
                                animal_ids=animal_ids,
                            )
                            metrics["roc_auc_lower"], metrics["roc_auc_upper"] = ci_roc
                            metrics["pr_auc_lower"], metrics["pr_auc_upper"] = ci_pr
                            
                            ci_brier = bootstrap_ci(
                                y_test.values,
                                y_pred,
                                brier_score_loss,
                                y_score=y_score,
                                n_bootstraps=bootstrap_n,
                                animal_ids=animal_ids,
                            )
                            metrics["brier_lower"], metrics["brier_upper"] = ci_brier
                            
                            ci_ece = bootstrap_ci(
                                y_test.values,
                                y_pred,
                                expected_calibration_error,
                                y_score=y_score,
                                n_bootstraps=bootstrap_n,
                                animal_ids=animal_ids,
                            )
                            metrics["ece_lower"], metrics["ece_upper"] = ci_ece
                    else:
                        if is_hgb:
                            model = model_class(
                                max_iter=100,
                                max_depth=6,
                                learning_rate=0.1,
                                min_samples_leaf=10,
                                random_state=RANDOM_STATE,
                                categorical_features=cat_indices if cat_indices else None,
                            )
                        else:
                            model = model_class(
                                iterations=100,
                                depth=6,
                                learning_rate=0.1,
                                verbose=0,
                                random_state=RANDOM_STATE,
                            )
                        
                        if is_hgb:
                            model.fit(X_tr, y_train)
                        else:
                            model.fit(X_tr, y_train, cat_features=cat_indices if cat_indices else None)
                        
                        y_pred = model.predict(X_te)
                        metrics = regression_metrics(y_test, y_pred, compute_ci=False)
                        
                        if compute_ci:
                            animal_ids = test_df["animal_id"].values if "animal_id" in test_df.columns else None
                            ci_mae = bootstrap_ci(
                                y_test.values,
                                y_pred,
                                mean_absolute_error,
                                n_bootstraps=bootstrap_n,
                                animal_ids=animal_ids,
                            )
                            metrics["mae_lower"], metrics["mae_upper"] = ci_mae
                            
                            ci_rmse = bootstrap_ci(
                                y_test.values,
                                y_pred,
                                lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
                                n_bootstraps=bootstrap_n,
                                animal_ids=animal_ids,
                            )
                            metrics["rmse_lower"], metrics["rmse_upper"] = ci_rmse
                            
                            ci_r2 = bootstrap_ci(
                                y_test.values,
                                y_pred,
                                r2_score,
                                n_bootstraps=bootstrap_n,
                                animal_ids=animal_ids,
                            )
                            metrics["r2_lower"], metrics["r2_upper"] = ci_r2
                    
                    result = {
                        "train_years": train_period,
                        "test_year": test_year,
                        "subset": subset_name,
                        "animal_subset": subset_name,
                        "model": model_name,
                        "model_name": model_name,
                        "pr_auc": metrics.get("pr_auc"),
                        "roc_auc": metrics.get("roc_auc"),
                        "brier": metrics.get("brier_score"),
                        "brier_score": metrics.get("brier_score"),
                        "ece": metrics.get("expected_calibration_error"),
                        "mae": metrics.get("mae"),
                        "rmse": metrics.get("rmse"),
                        "r2": metrics.get("r2"),
                        "train_rows": len(X_train),
                        "test_rows": len(X_test),
                    }
                    
                    if "pr_auc_lower" in metrics:
                        result["pr_auc_lower"] = metrics["pr_auc_lower"]
                        result["pr_auc_upper"] = metrics["pr_auc_upper"]
                    if "roc_auc_lower" in metrics:
                        result["roc_auc_lower"] = metrics["roc_auc_lower"]
                        result["roc_auc_upper"] = metrics["roc_auc_upper"]
                    if "brier_lower" in metrics:
                        result["brier_lower"] = metrics["brier_lower"]
                        result["brier_upper"] = metrics["brier_upper"]
                    if "ece_lower" in metrics:
                        result["ece_lower"] = metrics["ece_lower"]
                        result["ece_upper"] = metrics["ece_upper"]
                    if "mae_lower" in metrics:
                        result["mae_lower"] = metrics["mae_lower"]
                        result["mae_upper"] = metrics["mae_upper"]
                    if "rmse_lower" in metrics:
                        result["rmse_lower"] = metrics["rmse_lower"]
                        result["rmse_upper"] = metrics["rmse_upper"]
                    if "r2_lower" in metrics:
                        result["r2_lower"] = metrics["r2_lower"]
                        result["r2_upper"] = metrics["r2_upper"]
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error training {model_name} on {subset_name} for test year {test_year}: {e}", exc_info=True)
                    if strict:
                        raise e
    
    results_df = pd.DataFrame(results)
    
    # Coerce metric columns to numeric so they have float dtypes matching CSV loaded data
    metric_cols = ["pr_auc", "roc_auc", "brier", "brier_score", "ece", "mae", "rmse", "r2"]
    for col in metric_cols:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
            
    if output_path and not results_df.empty:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
    
    return results_df


def _train_classifier(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    compute_ci: bool = True,
) -> dict:
    """Train and evaluate a classifier."""
    if model_type == "catboost":
        model = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            verbose=0,
            random_state=RANDOM_STATE,
            auto_class_weights="Balanced",
        )
        model.fit(X_train, y_train, cat_features=[])
        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
    elif model_type == "histgradientboosting":
        model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=6,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    metrics = classification_metrics(y_test, y_pred, y_score=y_score, compute_ci=compute_ci)
    metrics["model_type"] = model_type
    return metrics


def _train_regressor(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    compute_ci: bool = True,
) -> dict:
    """Train and evaluate a regressor."""
    if model_type == "catboost":
        model = CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            verbose=0,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train, cat_features=[])
        y_pred = model.predict(X_test)
    elif model_type == "histgradientboosting":
        model = HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=6,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    metrics = regression_metrics(y_test, y_pred, compute_ci=compute_ci)
    metrics["model_type"] = model_type
    return metrics

