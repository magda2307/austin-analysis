"""Horizon-based yearly backtesting for adoption targets (7/30/60/90 days)."""

from pathlib import Path
import logging
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from aac_adoption.config import RANDOM_STATE
from aac_adoption.models.evaluate import classification_metrics_with_ci

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HORIZON_DAYS = [7, 30, 60, 90]
HORIZON_COLUMNS = {
    7: "adopted_in_7d",
    30: "adopted_in_30d",
    60: "adopted_in_60d",
    90: "adopted_in_90d",
}

FEATURE_CATEGORIES = {
    "categorical": [
        "animal_type", "intake_type", "intake_condition", "sex_upon_intake",
        "breed", "color", "is_austin_found_location", "is_outside_jurisdiction",
        "is_intersection_location", "is_address_like_location", "is_airport_location",
        "is_named", "age_group", "intake_month", "intake_quarter", "intake_season",
        "covid_period", "color_group", "primary_color", "simplified_color_group",
        "is_black_or_dark", "primary_breed", "is_mixed_breed", "simplified_breed_group"
    ],
    "numeric": ["age_days", "age_upon_intake", "followup_days_available"],
}


def prepare_features(df, feature_cols, cat_features):
    """Prepare feature matrix with proper handling of categoricals and NaNs."""
    X = df[feature_cols].copy()
    
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown").astype(str)
    
    numeric_cols = [c for c in feature_cols if c not in cat_features]
    for col in numeric_cols:
        if col in X.columns:
            X[col] = X[col].fillna(0)
    
    return X


def run_horizon_backtesting():
    """Run horizon-based yearly backtesting for allHorizon targets.
    
    Returns DataFrame with results for each horizon+year+model combination.
    """
    data_path = Path("data/processed/modeling_dataset.csv")
    if not data_path.exists():
        logger.error(f"Missing {data_path}")
        return None
    
    logger.info("Loading dataset...")
    df = pd.read_csv(data_path)
    
    feature_cols = []
    for cat in FEATURE_CATEGORIES.values():
        feature_cols.extend([c for c in cat if c in df.columns])
    feature_cols = list(dict.fromkeys(feature_cols))
    
    cat_features = [c for c in FEATURE_CATEGORIES["categorical"] if c in df.columns]
    
    results = []
    
    logger.info("Identifying test years...")
    all_years = sorted(df["intake_year"].dropna().unique())
    min_year = int(all_years[0])
    max_year = int(all_years[-1])
    test_years = list(range(min_year + 1, max_year + 1))
    
    logger.info(f"Test years: {test_years}")
    
    main_output_path = Path("reports/tables/yearly_backtesting.csv")
    if main_output_path.exists():
        existing_df = pd.read_csv(main_output_path)
        existing_cols = set(existing_df.columns)
    else:
        existing_df = pd.DataFrame()
        existing_cols = set()
    
    horizon_output_path = Path("reports/tables/horizon_yearly_backtesting.csv")
    
    for horizon_days in HORIZON_DAYS:
        target_col = HORIZON_COLUMNS[horizon_days]
        logger.info(f"\n--- Processing horizon: {horizon_days} days ---")
        
        for test_year in test_years:
            logger.info(f"  Test year: {test_year}")
            
            for min_train_year in range(min_year, test_year):
                if min_train_year == min_year:
                    train_years_str = f"{min_year}-{test_year - 1}"
                else:
                    train_years_str = f"{min_train_year}-{test_year - 1}"
                
                train_mask = (df["intake_year"] >= min_train_year) & \
                            (df["intake_year"] < test_year) & \
                            (df["followup_days_available"] >= horizon_days)
                test_mask = (df["intake_year"] == test_year) & \
                           (df["followup_days_available"] >= horizon_days)
                
                if train_mask.sum() < 100 or test_mask.sum() < 20:
                    continue
                
                df_train = df[train_mask].copy()
                df_test = df[test_mask].copy()
                
                X_train = prepare_features(df_train, feature_cols, cat_features)
                X_test = prepare_features(df_test, feature_cols, cat_features)
                
                y_train = df_train[target_col].values
                y_test = df_test[target_col].values
                
                model_results = []
                
                for model_name, model in [
                    ("CatBoost", CatBoostClassifier(
                        iterations=300,
                        auto_class_weights="Balanced",
                        cat_features=cat_features,
                        verbose=0,
                        random_seed=RANDOM_STATE
                    )),
                    ("HistGradientBoosting", HistGradientBoostingClassifier(
                        max_iter=300,
                        random_state=RANDOM_STATE
                    )),
                ]:
                    logger.info(f"    Training {model_name}...")
                    try:
                        model.fit(X_train.values, y_train)
                        y_pred_proba = model.predict_proba(X_test.values)[:, 1]
                        
                        metrics = classification_metrics_with_ci(
                            y_test, model.predict(X_test.values), y_pred_proba, n_bootstraps=1000
                        )
                        
                        model_results.append({
                            "horizon_days": horizon_days,
                            "train_years": train_years_str,
                            "test_year": test_year,
                            "model": model_name,
                            "train_rows": int(len(df_train)),
                            "test_rows": int(len(df_test)),
                            "min_followup_days": horizon_days,
                            "pr_auc": metrics["pr_auc"],
                            "pr_auc_lower": metrics["pr_auc_lower"],
                            "pr_auc_upper": metrics["pr_auc_upper"],
                            "roc_auc": metrics["roc_auc"],
                            "roc_auc_lower": metrics["roc_auc_lower"],
                            "roc_auc_upper": metrics["roc_auc_upper"],
                            "brier_score": metrics["brier_score"],
                            "brier_lower": metrics["brier_lower"],
                            "brier_upper": metrics["brier_upper"],
                            "expected_calibration_error": metrics["expected_calibration_error"],
                            "ece_lower": metrics["ece_lower"],
                            "ece_upper": metrics["ece_upper"],
                        })
                        
                        logger.info(f"      PR-AUC: {metrics['pr_auc']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
                        
                    except Exception as e:
                        logger.error(f"      Error training {model_name}: {e}")
                        continue
                
                results.extend(model_results)
    
    if not results:
        logger.warning("No results generated!")
        return None
    
    results_df = pd.DataFrame(results)
    
    output_path = Path("reports/tables/horizon_yearly_backtesting.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved horizon results to {output_path}")
    
    combined_path = Path("reports/tables/yearly_backtesting.csv")
    if not combined_path.exists() or existing_cols != set(results_df.columns):
        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
        combined_df.to_csv(combined_path, index=False)
        logger.info(f"Appended to main backtesting CSV: {combined_path}")
    else:
        logger.info(f"Results compatible with existing format")
    
    return results_df


if __name__ == "__main__":
    results = run_horizon_backtesting()
    if results is not None:
        print("\nSample output rows:")
        print(results.head(10).to_string())
        print(f"\nTotal horizon-year combinations tested: {len(results)}")
