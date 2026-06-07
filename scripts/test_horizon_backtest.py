"""Test horizon-based backtesting."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
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
from aac_adoption.models.evaluate import classification_metrics

TRAIN_YEARS = [2013, 2014, 2015, 2016, 2017, 2018]
ANIMAL_SUBSETS = ["combined", "dogs", "cats"]
HORIZON_DAYS = [7, 30, 60, 90]


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


def prepare_horizon_data(df: pd.DataFrame, horizon_days: int, years: list[int]) -> pd.DataFrame:
    """Prepare data for horizon target, filtering by sufficient follow-up time."""
    subset_df = df[df["intake_year"].isin(years)]
    subset_df = subset_df[subset_df["followup_days_available"] >= horizon_days].copy()
    horizon_col = f"adopted_in_{horizon_days}d"
    subset_df = subset_df.dropna(subset=[horizon_col])
    return subset_df


def get_test_data_horizon(df: pd.DataFrame, horizon_days: int, test_year: int) -> pd.DataFrame:
    """Get test data for specific year with sufficient follow-up time."""
    test_df = df[df["intake_year"] == test_year]
    test_df = test_df[test_df["followup_days_available"] >= horizon_days].copy()
    horizon_col = f"adopted_in_{horizon_days}d"
    test_df = test_df.dropna(subset=[horizon_col])
    return test_df


def run_horizon_backtesting(
    data_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Run yearly backtesting with horizon-based targets (7/30/60/90 days)."""
    header = pd.read_csv(data_path, nrows=0)
    parse_dates = [col for col in ["intake_datetime", "outcome_datetime"] if col in header.columns]
    df = pd.read_csv(data_path, parse_dates=parse_dates)
    
    results = []
    
    for subset in ANIMAL_SUBSETS:
        for horizon_days in HORIZON_DAYS:
            train_years = TRAIN_YEARS
            for i in range(len(TRAIN_YEARS)):
                test_year = 2019 + i
                
                for target_column in [f"adopted_in_{horizon_days}d"]:
                    train_df = prepare_horizon_data(df, horizon_days, train_years)
                    if subset.lower() == "dogs":
                        train_df = train_df[train_df["animal_type"].astype(str).str.lower() == "dog"]
                    elif subset.lower() == "cats":
                        train_df = train_df[train_df["animal_type"].astype(str).str.lower() == "cat"]
                    
                    train_df = train_df.dropna(subset=[target_column])
                    
                    test_df = get_test_data_horizon(df, horizon_days, test_year)
                    if subset.lower() == "dogs":
                        test_df = test_df[test_df["animal_type"].astype(str).str.lower() == "dog"]
                    elif subset.lower() == "cats":
                        test_df = test_df[test_df["animal_type"].astype(str).str.lower() == "cat"]
                    
                    test_df = test_df.dropna(subset=[target_column])
                    
                    if train_df.empty or test_df.empty:
                        print(f"Skipping {subset}/{horizon_days}d/{test_year}: insufficient data")
                        continue
                    
                    feature_columns = model_feature_columns(train_df)
                    if not feature_columns:
                        print(f"Skipping {subset}/{horizon_days}d/{test_year}: no features")
                        continue
                    
                    catboost_model = CatBoostClassifier(
                        auto_class_weights="Balanced",
                        random_state=RANDOM_STATE,
                        verbose=0,
                    )
                    hist_model = HistGradientBoostingClassifier(
                        class_weight="balanced",
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
                    
                    fit_params = {}
                    if "sample_weight" in train_df.columns:
                        fit_params["model__sample_weight"] = train_df["sample_weight"]
                    
                    catboost_pipeline.fit(train_df[feature_columns], train_df[target_column], **fit_params)
                    hist_pipeline.fit(train_df[feature_columns], train_df[target_column])
                    
                    catboost_preds = catboost_pipeline.predict(test_df[feature_columns])
                    hist_preds = hist_pipeline.predict(test_df[feature_columns])
                    
                    catboost_proba = catboost_pipeline.predict_proba(test_df[feature_columns])
                    hist_proba = hist_pipeline.predict_proba(test_df[feature_columns])
                    
                    catboost_score = None
                    hist_score = None
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
                    
                    results.append({
                        "train_years": f"{train_years[0]}-{train_years[-1]}",
                        "test_year": test_year,
                        "subset": subset,
                        "model": "catboost_classifier",
                        "horizon_days": horizon_days,
                        "train_rows": len(train_df),
                        "test_rows": len(test_df),
                        "min_followup_days": horizon_days,
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
                        "model": "hist_gradient_classifier",
                        "horizon_days": horizon_days,
                        "train_rows": len(train_df),
                        "test_rows": len(test_df),
                        "min_followup_days": horizon_days,
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
                        "ece_lower": None,
                        "ece_upper": None,
                        "mae_lower": None,
                        "mae_upper": None,
                        "rmse_lower": None,
                        "rmse_upper": None,
                        "r2_lower": None,
                        "r2_upper": None,
                    })
    
    results_df = pd.DataFrame(results)
    
    existing_df = pd.DataFrame()
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
    
    combined_df = pd.concat([existing_df, results_df], ignore_index=True) if not existing_df.empty else results_df
    
    combined_df.to_csv(output_path, index=False)
    
    return results_df


if __name__ == "__main__":
    data_path = Path("data/processed/modeling_dataset.csv")
    output_path = Path("reports/tables/yearly_backtesting.csv")
    
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        sys.exit(1)
    
    print("Running horizon-based yearly backtesting...")
    horizon_results_df = run_horizon_backtesting(data_path, output_path)
    print(f"Horizon results saved to {output_path}")
    print(f"\nHorizon Summary:")
    print(f"- Total horizon records: {len(horizon_results_df)}")
    print(f"- Models: {horizon_results_df['model'].unique().tolist()}")
    print(f"- Subsets: {horizon_results_df['subset'].unique().tolist()}")
    print(f"- Horizon days: {sorted(horizon_results_df['horizon_days'].unique().tolist())}")
    print(f"- Test years: {sorted(horizon_results_df['test_year'].unique().tolist())}")
    
    print("\nSample output rows:")
    sample_rows = horizon_results_df.head(6)
    print(sample_rows.to_string(index=False))
