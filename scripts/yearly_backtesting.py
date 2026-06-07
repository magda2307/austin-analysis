"""Yearly temporal backtesting for slice 13.

Uses train periods 2013-2018, 2013-2019, 2013-2020, 2013-2021, 2013-2022, 2013-2023
and tests on corresponding next years: 2019, 2020, 2021, 2022, 2023, 2024.

Tests multiple models:
- CatBoostClassifier (auto_class_weights="Balanced")
- CatBoostRegressor
- HistGradientBoostingClassifier (class_weight="balanced")
- HistGradientBoostingRegressor

Metrics per model/year:
- Classification: PR-AUC, ROC-AUC, Brier score, ECE
- Regression: MAE, RMSE, R²

Cluster-aware bootstrap confidence intervals (95%) for key metrics.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from aac_adoption.config import RANDOM_STATE
from aac_adoption.features.feature_sets import (
    model_feature_columns,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from aac_adoption.models.evaluate import (
    classification_metrics,
    regression_metrics,
    expected_calibration_error,
    bootstrap_ci,
)
from aac_adoption.models.split import make_time_split

TRAIN_START = 2013
TRAIN_ENDS = [2018, 2019, 2020, 2021, 2022, 2023]
TEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
ANIMAL_SUBSETS = ["combined", "dogs", "cats"]


def categorical_to_object(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype("object")


def make_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [
        col for col in NUMERIC_FEATURES if col in df.columns
    ]
    categorical_features = [
        col for col in CATEGORICAL_FEATURES if col in df.columns
    ]
    
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
    
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0))
    ])
    categorical_pipeline = Pipeline(steps=[
        ("as_object", FunctionTransformer(categorical_to_object, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
    ])
    
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        sparse_threshold=0.0,
    )


def prepare_data(df: pd.DataFrame, target_column: str, train_years: list[int]) -> pd.DataFrame:
    subset_df = df[df["intake_year"].isin(train_years)]
    subset_df = subset_df.dropna(subset=[target_column])
    return subset_df


def get_test_data(df: pd.DataFrame, target_column: str, test_year: int) -> pd.DataFrame:
    test_df = df[df["intake_year"] == test_year]
    test_df = test_df.dropna(subset=[target_column])
    return test_df


def compute_classification_metrics_with_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    animal_ids: np.ndarray | None,
) -> dict:
    """Compute classification metrics with cluster-aware bootstrap CI."""
    base_metrics = classification_metrics(y_true, y_pred, y_score, compute_ci=False)
    
    if animal_ids is not None:
        animal_ids_arr = np.asarray(animal_ids)
    else:
        animal_ids_arr = None
    
    base_metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    base_metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    base_metrics["brier_score"] = float(brier_score_loss(y_true, y_score))
    base_metrics["expected_calibration_error"] = expected_calibration_error(y_true, y_score)
    
    if animal_ids_arr is not None and len(np.unique(animal_ids_arr)) >= 2:
        ci_pr = bootstrap_ci(
            y_true, y_pred, average_precision_score,
            y_score=y_score, n_bootstraps=1000, random_state=RANDOM_STATE,
            animal_ids=animal_ids_arr
        )
        ci_roc = bootstrap_ci(
            y_true, y_pred, roc_auc_score,
            y_score=y_score, n_bootstraps=1000, random_state=RANDOM_STATE,
            animal_ids=animal_ids_arr
        )
        ci_brier = bootstrap_ci(
            y_true, y_pred, brier_score_loss,
            y_score=y_score, n_bootstraps=1000, random_state=RANDOM_STATE,
            animal_ids=animal_ids_arr
        )
        ci_ece = bootstrap_ci(
            y_true, y_pred, expected_calibration_error,
            y_score=y_score, n_bootstraps=1000, random_state=RANDOM_STATE,
            animal_ids=animal_ids_arr
        )
        
        base_metrics["pr_auc_lower"], base_metrics["pr_auc_upper"] = ci_pr
        base_metrics["roc_auc_lower"], base_metrics["roc_auc_upper"] = ci_roc
        base_metrics["brier_lower"], base_metrics["brier_upper"] = ci_brier
        base_metrics["ece_lower"], base_metrics["ece_upper"] = ci_ece
    else:
        base_metrics["pr_auc_lower"] = base_metrics["pr_auc_upper"] = np.nan
        base_metrics["roc_auc_lower"] = base_metrics["roc_auc_upper"] = np.nan
        base_metrics["brier_lower"] = base_metrics["brier_upper"] = np.nan
        base_metrics["ece_lower"] = base_metrics["ece_upper"] = np.nan
    
    return base_metrics


def compute_regression_metrics_with_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    animal_ids: np.ndarray | None,
) -> dict:
    """Compute regression metrics with cluster-aware bootstrap CI."""
    base_metrics = regression_metrics(y_true, y_pred, compute_ci=False)
    
    if animal_ids is not None:
        animal_ids_arr = np.asarray(animal_ids)
    else:
        animal_ids_arr = None
    
    if animal_ids_arr is not None and len(np.unique(animal_ids_arr)) >= 2:
        ci_mae = bootstrap_ci(
            y_true, y_pred, lambda yt, yp: np.mean(np.abs(yt - yp)),
            n_bootstraps=1000, random_state=RANDOM_STATE,
            animal_ids=animal_ids_arr
        )
        ci_rmse = bootstrap_ci(
            y_true, y_pred, lambda yt, yp: np.sqrt(np.mean((yt - yp) ** 2)),
            n_bootstraps=1000, random_state=RANDOM_STATE,
            animal_ids=animal_ids_arr
        )
        ci_r2 = bootstrap_ci(
            y_true, y_pred, lambda yt, yp: 1 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2),
            n_bootstraps=1000, random_state=RANDOM_STATE,
            animal_ids=animal_ids_arr
        )
        
        base_metrics["mae_lower"], base_metrics["mae_upper"] = ci_mae
        base_metrics["rmse_lower"], base_metrics["rmse_upper"] = ci_rmse
        base_metrics["r2_lower"], base_metrics["r2_upper"] = ci_r2
    else:
        base_metrics["mae_lower"] = base_metrics["mae_upper"] = np.nan
        base_metrics["rmse_lower"] = base_metrics["rmse_upper"] = np.nan
        base_metrics["r2_lower"] = base_metrics["r2_upper"] = np.nan
    
    return base_metrics


def run_yearly_backtesting(
    data_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    header = pd.read_csv(data_path, nrows=0)
    parse_dates = [col for col in ["intake_datetime", "outcome_datetime"] if col in header.columns]
    df = pd.read_csv(data_path, parse_dates=parse_dates)
    
    results = []
    
    for target_type, target_column in [("classification", "classification_target"), ("regression", "regression_target_days")]:
        for subset in ANIMAL_SUBSETS:
            for i, train_end in enumerate(TRAIN_ENDS):
                train_years = list(range(TRAIN_START, train_end + 1))
                test_year = TEST_YEARS[i]
                
                train_df = prepare_data(df, target_column, train_years)
                test_df = get_test_data(df, target_column, test_year)
                
                if train_df.empty or test_df.empty:
                    continue
                
                if subset != "combined":
                    subset_lower = subset.lower()
                    train_df = train_df[train_df["animal_type"].astype(str).str.lower() == subset_lower]
                    test_df = test_df[test_df["animal_type"].astype(str).str.lower() == subset_lower]
                
                if train_df.empty or test_df.empty:
                    continue
                
                feature_columns = model_feature_columns(train_df)
                if not feature_columns:
                    continue
                
                preprocessor = make_preprocessor(train_df[feature_columns])
                
                if target_type == "classification":
                    catboost_model = CatBoostClassifier(
                        auto_class_weights="Balanced",
                        random_state=RANDOM_STATE,
                        verbose=0,
                    )
                    hist_model = HistGradientBoostingClassifier(
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    )
                else:
                    catboost_model = CatBoostRegressor(
                        random_state=RANDOM_STATE,
                        verbose=0,
                    )
                    hist_model = HistGradientBoostingRegressor(
                        random_state=RANDOM_STATE,
                    )
                
                catboost_pipeline = Pipeline(steps=[
                    ("preprocess", preprocessor),
                    ("model", catboost_model),
                ])
                hist_pipeline = Pipeline(steps=[
                    ("preprocess", preprocessor),
                    ("model", hist_model),
                ])
                
                catboost_pipeline.fit(train_df[feature_columns], train_df[target_column])
                hist_pipeline.fit(train_df[feature_columns], train_df[target_column])
                
                catboost_preds = catboost_pipeline.predict(test_df[feature_columns])
                hist_preds = hist_pipeline.predict(test_df[feature_columns])
                
                if target_type == "classification":
                    catboost_proba = catboost_pipeline.predict_proba(test_df[feature_columns])
                    hist_proba = hist_pipeline.predict_proba(test_df[feature_columns])
                    
                    catboost_score = None
                    hist_score = None
                    if catboost_proba.shape[1] == 2:
                        catboost_score = catboost_proba[:, 1]
                    if hist_proba.shape[1] == 2:
                        hist_score = hist_proba[:, 1]
                    
                    if test_df["animal_id"].dtype == "object":
                        animal_ids = test_df["animal_id"].values
                    else:
                        animal_ids = test_df["animal_id"].astype(str).values
                    
                    catboost_metrics = compute_classification_metrics_with_bootstrap(
                        test_df[target_column].values,
                        catboost_preds,
                        catboost_score,
                        animal_ids if catboost_score is not None else None
                    )
                    
                    hist_metrics = compute_classification_metrics_with_bootstrap(
                        test_df[target_column].values,
                        hist_preds,
                        hist_score,
                        animal_ids if hist_score is not None else None
                    )
                    
                    results.append({
                        "train_years": f"{train_years[0]}-{train_years[-1]}",
                        "test_year": test_year,
                        "subset": subset,
                        "model": "catboost_classifier",
                        "pr_auc": catboost_metrics.get("pr_auc"),
                        "roc_auc": catboost_metrics.get("roc_auc"),
                        "brier": catboost_metrics.get("brier_score"),
                        "ece": catboost_metrics.get("expected_calibration_error"),
                        "mae": None,
                        "rmse": None,
                        "r2": None,
                        "pr_auc_lower": catboost_metrics.get("pr_auc_lower"),
                        "pr_auc_upper": catboost_metrics.get("pr_auc_upper"),
                        "roc_auc_lower": catboost_metrics.get("roc_auc_lower"),
                        "roc_auc_upper": catboost_metrics.get("roc_auc_upper"),
                        "brier_lower": catboost_metrics.get("brier_lower"),
                        "brier_upper": catboost_metrics.get("brier_upper"),
                        "ece_lower": catboost_metrics.get("ece_lower"),
                        "ece_upper": catboost_metrics.get("ece_upper"),
                        "mae_lower": None,
                        "mae_upper": None,
                        "rmse_lower": None,
                        "rmse_upper": None,
                        "r2_lower": None,
                        "r2_upper": None,
                    })
                    
                    results.append({
                        "train_years": f"{train_years[0]}-{train_years[-1]}",
                        "test_year": test_year,
                        "subset": subset,
                        "model": "hist_gradient_classifier",
                        "pr_auc": hist_metrics.get("pr_auc"),
                        "roc_auc": hist_metrics.get("roc_auc"),
                        "brier": hist_metrics.get("brier_score"),
                        "ece": hist_metrics.get("expected_calibration_error"),
                        "mae": None,
                        "rmse": None,
                        "r2": None,
                        "pr_auc_lower": hist_metrics.get("pr_auc_lower"),
                        "pr_auc_upper": hist_metrics.get("pr_auc_upper"),
                        "roc_auc_lower": hist_metrics.get("roc_auc_lower"),
                        "roc_auc_upper": hist_metrics.get("roc_auc_upper"),
                        "brier_lower": hist_metrics.get("brier_lower"),
                        "brier_upper": hist_metrics.get("brier_upper"),
                        "ece_lower": hist_metrics.get("ece_lower"),
                        "ece_upper": hist_metrics.get("ece_upper"),
                        "mae_lower": None,
                        "mae_upper": None,
                        "rmse_lower": None,
                        "rmse_upper": None,
                        "r2_lower": None,
                        "r2_upper": None,
                    })
                
                else:
                    if test_df["animal_id"].dtype == "object":
                        animal_ids = test_df["animal_id"].values
                    else:
                        animal_ids = test_df["animal_id"].astype(str).values
                    
                    catboost_metrics = compute_regression_metrics_with_bootstrap(
                        test_df[target_column].values,
                        catboost_preds,
                        animal_ids
                    )
                    
                    hist_metrics = compute_regression_metrics_with_bootstrap(
                        test_df[target_column].values,
                        hist_preds,
                        animal_ids
                    )
                    
                    results.append({
                        "train_years": f"{train_years[0]}-{train_years[-1]}",
                        "test_year": test_year,
                        "subset": subset,
                        "model": "catboost_regressor",
                        "pr_auc": None,
                        "roc_auc": None,
                        "brier": None,
                        "ece": None,
                        "mae": catboost_metrics.get("mae"),
                        "rmse": catboost_metrics.get("rmse"),
                        "r2": catboost_metrics.get("r2"),
                        "pr_auc_lower": None,
                        "pr_auc_upper": None,
                        "roc_auc_lower": None,
                        "roc_auc_upper": None,
                        "brier_lower": None,
                        "brier_upper": None,
                        "ece_lower": None,
                        "ece_upper": None,
                        "mae_lower": catboost_metrics.get("mae_lower"),
                        "mae_upper": catboost_metrics.get("mae_upper"),
                        "rmse_lower": catboost_metrics.get("rmse_lower"),
                        "rmse_upper": catboost_metrics.get("rmse_upper"),
                        "r2_lower": catboost_metrics.get("r2_lower"),
                        "r2_upper": catboost_metrics.get("r2_upper"),
                    })
                    
                    results.append({
                        "train_years": f"{train_years[0]}-{train_years[-1]}",
                        "test_year": test_year,
                        "subset": subset,
                        "model": "hist_gradient_regressor",
                        "pr_auc": None,
                        "roc_auc": None,
                        "brier": None,
                        "ece": None,
                        "mae": hist_metrics.get("mae"),
                        "rmse": hist_metrics.get("rmse"),
                        "r2": hist_metrics.get("r2"),
                        "pr_auc_lower": None,
                        "pr_auc_upper": None,
                        "roc_auc_lower": None,
                        "roc_auc_upper": None,
                        "brier_lower": None,
                        "brier_upper": None,
                        "ece_lower": None,
                        "ece_upper": None,
                        "mae_lower": hist_metrics.get("mae_lower"),
                        "mae_upper": hist_metrics.get("mae_upper"),
                        "rmse_lower": hist_metrics.get("rmse_lower"),
                        "rmse_upper": hist_metrics.get("rmse_upper"),
                        "r2_lower": hist_metrics.get("r2_lower"),
                        "r2_upper": hist_metrics.get("r2_upper"),
                    })
    
    results_df = pd.DataFrame(results)
    
    results_df.to_csv(output_path, index=False)
    
    return results_df


def main() -> None:
    data_path = Path("data/processed/modeling_dataset.csv")
    output_path = Path("reports/tables/yearly_backtesting.csv")
    
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        sys.exit(1)
    
    print("Running yearly temporal backtesting...")
    results_df = run_yearly_backtesting(data_path, output_path)
    print(f"Results saved to {output_path}")
    print(f"\nSummary:")
    print(f"- Total records: {len(results_df)}")
    print(f"- Models: {results_df['model'].unique().tolist()}")
    print(f"- Subsets: {results_df['subset'].unique().tolist()}")
    print(f"- Train year ranges: {results_df['train_years'].unique().tolist()}")
    print(f"- Test years: {sorted(results_df['test_year'].unique().tolist())}")


if __name__ == "__main__":
    main()
