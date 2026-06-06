"""Yearly temporal backtesting using rolling historical windows."""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from aac_adoption.config import RANDOM_STATE
from aac_adoption.models.evaluate import bootstrap_ci
from aac_adoption.models.evaluate import classification_metrics, regression_metrics
from aac_adoption.models.split import make_time_split, _filter_subset


TRAIN_START_YEAR = 2013


def get_test_years(df: pd.DataFrame) -> List[int]:
    """Get unique test years from dataset."""
    years = sorted(df["intake_year"].dropna().astype(int).unique())
    return years


def get_train_years(test_year: int, max_train_year: Optional[int] = None) -> Tuple[int, int]:
    """Get training window (2013-X) for given test year."""
    end_year = min(test_year - 1, max_train_year) if max_train_year else test_year - 1
    return (TRAIN_START_YEAR, end_year)


def format_train_period(start: int, end: int) -> str:
    """Format training period string."""
    return f"{start}-{end}"


def _detect_categorical_features(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> list[str]:
    """Detect categorical columns in DataFrame."""
    exclude_cols = exclude_cols or []
    TARGET_KEYWORDS = ["target", "label", "outcome", "class", "y_", "days", "datetime", "date", "time"]
    categorical_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        # Skip columns that look like targets or datetime
        if any(kw in col.lower() for kw in TARGET_KEYWORDS):
            continue
        if df[col].dtype == "object" or df[col].dtype.name == "string":
            categorical_cols.append(col)
    return categorical_cols


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


def run_yearly_backtesting(
    df: pd.DataFrame,
    target_column: str,
    animal_subset: str = "combined",
    output_path: Optional[str] = None,
    compute_ci: bool = True,
    bootstrap_n: int = 100,
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
        
    Returns:
        DataFrame with one row per train_window/test_year combination
    """
    subset_df, subset_name = _filter_subset(df, animal_subset)
    subset_df = subset_df.dropna(subset=[target_column]).copy()
    
    test_years = get_test_years(subset_df)
    results = []
    
    for test_year in test_years:
        train_start, train_end = get_train_years(test_year)
        train_period = format_train_period(train_start, train_end)
        
        train_mask = subset_df["intake_year"].between(train_start, train_end)
        test_mask = subset_df["intake_year"] == test_year
        
        train_df = subset_df[train_mask].copy()
        test_df = subset_df[test_mask].copy()
        
        if train_df.empty or test_df.empty:
            continue
        
        X_train = train_df.drop(columns=[target_column, "animal_id", "intake_year"])
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=[target_column, "animal_id", "intake_year"])
        y_test = test_df[target_column]
        
        # Detect categorical features
        categorical_features = _detect_categorical_features(X_train, exclude_cols=[])
        cat_indices = [X_train.columns.get_loc(c) for c in categorical_features if c in X_train.columns]
        
        is_classification = len(y_train.unique()) <= 2 and y_train.dtype in [np.int64, np.int32, bool]
        
        if is_classification:
            models_to_test = [
                ("catboost_classifier", CatBoostClassifier),
                ("histgradientboosting_classifier", HistGradientBoostingClassifier),
            ]
        else:
            models_to_test = [
                ("catboost_regressor", CatBoostRegressor),
                ("histgradientboosting_regressor", HistGradientBoostingRegressor),
            ]
        
        for model_name, model_class in models_to_test:
            try:
                if model_name.endswith("_classifier"):
                    model = model_class(
                        iterations=100,
                        depth=6,
                        learning_rate=0.1,
                        verbose=0,
                        random_state=RANDOM_STATE,
                        auto_class_weights="Balanced",
                    )
                    model.fit(X_train, y_train, cat_features=cat_indices if cat_indices else None)
                    y_pred = model.predict(X_test)
                    if hasattr(model, "predict_proba"):
                        y_score = model.predict_proba(X_test)[:, 1]
                        metrics = classification_metrics(y_test, y_pred, y_score=y_score, compute_ci=False)
                    else:
                        metrics = classification_metrics(y_test, y_pred, y_score=None, compute_ci=False)
                    
                    if compute_ci and len(y_test.unique()) == 2:
                        # Use animal_ids for cluster-aware bootstrap
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
                    model = model_class(
                        iterations=100,
                        depth=6,
                        learning_rate=0.1,
                        verbose=0,
                        random_state=RANDOM_STATE,
                    )
                    model.fit(X_train, y_train, cat_features=cat_indices if cat_indices else None)
                    y_pred = model.predict(X_test)
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
                    "model": model_name,
                    "pr_auc": metrics.get("pr_auc"),
                    "roc_auc": metrics.get("roc_auc"),
                    "brier": metrics.get("brier_score"),
                    "ece": metrics.get("expected_calibration_error"),
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "r2": metrics.get("r2"),
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
                continue
    
    results_df = pd.DataFrame(results)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
    
    return results_df
