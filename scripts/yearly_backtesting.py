"""Yearly temporal backtesting for slice 13."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

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

TRAIN_YEARS = [2013, 2014, 2015, 2016, 2017, 2018]
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
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipeline = Pipeline(steps=[
        ("as_object", FunctionTransformer(categorical_to_object, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
    ])
    
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        sparse_threshold=0.0,
    )


def prepare_data(df: pd.DataFrame, target_column: str, years: list[int]) -> pd.DataFrame:
    subset_df = df[df["intake_year"].isin(years)]
    subset_df = subset_df.dropna(subset=[target_column])
    return subset_df


def get_test_data(df: pd.DataFrame, target_column: str, test_year: int) -> pd.DataFrame:
    test_df = df[df["intake_year"] == test_year]
    test_df = test_df.dropna(subset=[target_column])
    return test_df


def fit_and_evaluate_catboost(
    model_cls,
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    train_years: list[int],
    test_year: int,
    subset: str,
    model_params: dict,
) -> dict | None:
    train_df = prepare_data(df, target_column, train_years)
    test_df = get_test_data(df, target_column, test_year)
    
    if train_df.empty or test_df.empty:
        return None
    
    split = make_time_split(train_df, target_column, animal_subset=subset)
    if split.train.empty or (subset in ["dogs", "cats"] and split.test.empty):
        return None
    
    pipeline = Pipeline(steps=[
        ("preprocess", make_preprocessor(split.train[feature_columns])),
        ("model", model_cls(**model_params)),
    ])
    
    fit_params = {}
    if target_column == "classification_target" and "sample_weight" in split.train.columns:
        fit_params["model__sample_weight"] = split.train["sample_weight"]
    
    pipeline.fit(split.train[feature_columns], split.train[target_column], **fit_params)
    
    predictions = pipeline.predict(test_df[feature_columns])
    y_score = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(test_df[feature_columns])
        if proba.shape[1] == 2:
            y_score = proba[:, 1]
    
    metrics = classification_metrics(
        test_df[target_column],
        predictions,
        y_score,
        compute_ci=True
    )
    
    return metrics


def fit_and_evaluate_histgradient(
    model_cls,
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    train_years: list[int],
    test_year: int,
    subset: str,
    model_params: dict,
) -> dict | None:
    train_df = prepare_data(df, target_column, train_years)
    test_df = get_test_data(df, target_column, test_year)
    
    if train_df.empty or test_df.empty:
        return None
    
    split = make_time_split(train_df, target_column, animal_subset=subset)
    if split.train.empty or (subset in ["dogs", "cats"] and split.test.empty):
        return None
    
    pipeline = Pipeline(steps=[
        ("preprocess", make_preprocessor(split.train[feature_columns])),
        ("model", model_cls(**model_params)),
    ])
    
    pipeline.fit(split.train[feature_columns], split.train[target_column])
    
    predictions = pipeline.predict(test_df[feature_columns])
    y_score = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(test_df[feature_columns])
        if proba.shape[1] == 2:
            y_score = proba[:, 1]
    
    metrics = classification_metrics(
        test_df[target_column],
        predictions,
        y_score,
        compute_ci=True
    )
    
    return metrics


def run_yearly_backtesting(
    data_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    header = pd.read_csv(data_path, nrows=0)
    parse_dates = [col for col in ["intake_datetime", "outcome_datetime"] if col in header.columns]
    df = pd.read_csv(data_path, parse_dates=parse_dates)
    
    results = []
    
    for subset in ANIMAL_SUBSETS:
        for i, train_start in enumerate(TRAIN_YEARS):
            train_years = list(range(train_start, 2019))
            test_year = 2019 + i
            
            for target_type, target_column in [("classification", "classification_target"), ("regression", "regression_target_days")]:
                train_df = prepare_data(df, target_column, train_years)
                train_df_subset, _ = (subset.lower(), "combined") if subset.lower() == "combined" else (subset.lower(), subset.lower())
                if train_df_subset == "dogs":
                    train_df = train_df[train_df["animal_type"].astype(str).str.lower() == "dog"]
                elif train_df_subset == "cats":
                    train_df = train_df[train_df["animal_type"].astype(str).str.lower() == "cat"]
                
                train_df = train_df.dropna(subset=[target_column])
                
                test_df = get_test_data(df, target_column, test_year)
                if train_df_subset == "dogs":
                    test_df = test_df[test_df["animal_type"].astype(str).str.lower() == "dog"]
                elif train_df_subset == "cats":
                    test_df = test_df[test_df["animal_type"].astype(str).str.lower() == "cat"]
                
                test_df = test_df.dropna(subset=[target_column])
                
                if train_df.empty or test_df.empty:
                    continue
                
                feature_columns = model_feature_columns(train_df)
                if not feature_columns:
                    continue
                
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
                
                catboost_preprocessor = make_preprocessor(train_df[feature_columns])
                hist_preprocessor = make_preprocessor(train_df[feature_columns])
                
                catboost_pipeline = Pipeline(steps=[
                    ("preprocess", catboost_preprocessor),
                    ("model", catboost_model),
                ])
                hist_pipeline = Pipeline(steps=[
                    ("preprocess", hist_preprocessor),
                    ("model", hist_model),
                ])
                
                catboost_pipeline.fit(train_df[feature_columns], train_df[target_column])
                hist_pipeline.fit(train_df[feature_columns], train_df[target_column])
                
                catboost_preds = catboost_pipeline.predict(test_df[feature_columns])
                hist_preds = hist_pipeline.predict(test_df[feature_columns])
                
                catboost_score = None
                hist_score = None
                
                catboost_metrics = {}
                hist_metrics = {}
                catboost_metrics_reg = {}
                hist_metrics_reg = {}
                
                if target_type == "classification":
                    catboost_proba = catboost_pipeline.predict_proba(test_df[feature_columns])
                    hist_proba = hist_pipeline.predict_proba(test_df[feature_columns])
                    if catboost_proba.shape[1] == 2:
                        catboost_score = catboost_proba[:, 1]
                    if hist_proba.shape[1] == 2:
                        hist_score = hist_proba[:, 1]
                    
                    catboost_metrics = classification_metrics(
                        test_df[target_column],
                        catboost_preds,
                        catboost_score,
                        compute_ci=True
                    )
                    hist_metrics = classification_metrics(
                        test_df[target_column],
                        hist_preds,
                        hist_score,
                        compute_ci=True
                    )
                else:
                    catboost_metrics_reg = regression_metrics(
                        test_df[target_column],
                        catboost_preds,
                        compute_ci=True
                    )
                    hist_metrics_reg = regression_metrics(
                        test_df[target_column],
                        hist_preds,
                        compute_ci=True
                    )
                
                test_df["animal_id"] = test_df["animal_id"].astype(str)
                
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
                    "brier_lower": None,
                    "brier_upper": None,
                    "ece_lower": None,
                    "ece_upper": None,
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
                    "model": "catboost_regressor",
                    "pr_auc": None,
                    "roc_auc": None,
                    "brier": None,
                    "ece": None,
                    "mae": catboost_metrics_reg.get("mae"),
                    "rmse": catboost_metrics_reg.get("rmse"),
                    "r2": catboost_metrics_reg.get("r2"),
                    "pr_auc_lower": None,
                    "pr_auc_upper": None,
                    "roc_auc_lower": None,
                    "roc_auc_upper": None,
                    "brier_lower": None,
                    "brier_upper": None,
                    "ece_lower": None,
                    "ece_upper": None,
                    "mae_lower": catboost_metrics_reg.get("mae_lower"),
                    "mae_upper": catboost_metrics_reg.get("mae_upper"),
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
                    "brier_lower": None,
                    "brier_upper": None,
                    "ece_lower": None,
                    "ece_upper": None,
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
                    "model": "hist_gradient_regressor",
                    "pr_auc": None,
                    "roc_auc": None,
                    "brier": None,
                    "ece": None,
                    "mae": hist_metrics_reg.get("mae"),
                    "rmse": hist_metrics_reg.get("rmse"),
                    "r2": hist_metrics_reg.get("r2"),
                    "pr_auc_lower": None,
                    "pr_auc_upper": None,
                    "roc_auc_lower": None,
                    "roc_auc_upper": None,
                    "brier_lower": None,
                    "brier_upper": None,
                    "ece_lower": None,
                    "ece_upper": None,
                    "mae_lower": hist_metrics_reg.get("mae_lower"),
                    "mae_upper": hist_metrics_reg.get("mae_upper"),
                    "rmse_lower": None,
                    "rmse_upper": None,
                    "r2_lower": None,
                    "r2_upper": None,
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
