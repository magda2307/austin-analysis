"""Horizon-based yearly backtesting for adoption targets (7/30/60/90 days)."""

from pathlib import Path
import logging
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

CAT_FEATURES = [
    "animal_type", "intake_type", "intake_condition", "sex_upon_intake",
    "breed", "color", "is_austin_found_location", "is_outside_jurisdiction",
    "is_intersection_location", "is_address_like_location", "is_airport_location",
    "is_named", "age_group", "intake_month", "intake_quarter", "intake_season",
    "covid_period", "color_group", "primary_color", "simplified_color_group",
    "is_black_or_dark", "primary_breed", "is_mixed_breed", "simplified_breed_group"
]

NUM_FEATURES = ["age_days", "age_upon_intake", "followup_days_available"]


def encode_categorical_features(df, cat_features):
    """One-hot encode categorical features for HistGradientBoosting."""
    df_encoded = df.copy()
    
    for col in cat_features:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].fillna("Unknown").astype(str)
    
    df_dummies = pd.get_dummies(df_encoded, columns=cat_features, dummy_na=True, dtype=float)
    
    return df_dummies, list(df_dummies.columns)


def run_horizon_backtesting():
    """Run horizon-based yearly backtesting for all horizon targets.
    
    Train on all years before test_year (2013-X where X < test_year).
    Filter both train and test sets to include only intakes with sufficient follow-up time.
    
    Returns DataFrame with results for each horizon+year+model combination.
    """
    data_path = Path("data/processed/modeling_dataset.csv")
    if not data_path.exists():
        logger.error(f"Missing {data_path}")
        return None
    
    logger.info("Loading dataset...")
    df = pd.read_csv(data_path)
    
    results = []
    
    logger.info("Identifying test years...")
    all_years = sorted(df["intake_year"].dropna().unique())
    min_year = int(all_years[0])
    max_year = int(all_years[-1])
    test_years = list(range(min_year + 1, max_year + 1))
    
    logger.info(f"Test years: {test_years}")
    logger.info(f"Categorical features: {len(CAT_FEATURES)}")
    logger.info(f"Numeric features: {len(NUM_FEATURES)}")
    
    for horizon_days in HORIZON_DAYS:
        target_col = HORIZON_COLUMNS[horizon_days]
        logger.info(f"\n--- Processing horizon: {horizon_days} days ---")
        
        for test_year in test_years:
            logger.info(f"  Test year: {test_year}")
            
            train_mask = (df["intake_year"] < test_year) & \
                        (df["followup_days_available"] >= horizon_days)
            test_mask = (df["intake_year"] == test_year) & \
                       (df["followup_days_available"] >= horizon_days)
            
            if train_mask.sum() < 100 or test_mask.sum() < 20:
                continue
            
            df_train = df[train_mask].copy()
            df_test = df[test_mask].copy()
            
            y_train = df_train[target_col].values
            y_test = df_test[target_col].values
            
            train_years_str = f"{min_year}-{test_year - 1}"
            
            for model_name, is_cb in [
                ("CatBoost", True),
                ("HistGradientBoosting", False),
            ]:
                logger.info(f"    Training {model_name}...")
                try:
                    if is_cb:
                        X_train = df_train[CAT_FEATURES + NUM_FEATURES].copy()
                        X_test = df_test[CAT_FEATURES + NUM_FEATURES].copy()
                        
                        for col in CAT_FEATURES:
                            if col in X_train.columns:
                                X_train[col] = X_train[col].fillna("Unknown").astype(str)
                                X_test[col] = X_test[col].fillna("Unknown").astype(str)
                        
                        for col in NUM_FEATURES:
                            if col in X_train.columns:
                                X_train[col] = X_train[col].fillna(0)
                                X_test[col] = X_test[col].fillna(0)
                        
                        model = CatBoostClassifier(
                            iterations=300,
                            auto_class_weights="Balanced",
                            cat_features=CAT_FEATURES,
                            verbose=0,
                            random_seed=RANDOM_STATE
                        )
                        model.fit(X_train, y_train)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                    else:
                        X_train_full, feature_names = encode_categorical_features(
                            df_train[CAT_FEATURES + NUM_FEATURES], CAT_FEATURES
                        )
                        X_test_full, _ = encode_categorical_features(
                            df_test[CAT_FEATURES + NUM_FEATURES], CAT_FEATURES
                        )
                        
                        X_train = X_train_full.values
                        X_test = X_test_full.values
                        
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                        
                        model = HistGradientBoostingClassifier(
                            max_iter=300,
                            random_state=RANDOM_STATE
                        )
                        model.fit(X_train, y_train)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    metrics = classification_metrics_with_ci(
                        y_test, model.predict(X_test), y_pred_proba, n_bootstraps=1000
                    )
                    
                    results.append({
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
    
    if not results:
        logger.warning("No results generated!")
        return None
    
    results_df = pd.DataFrame(results)
    
    output_path = Path("reports/tables/horizon_yearly_backtesting.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved horizon results to {output_path}")
    
    main_output_path = Path("reports/tables/yearly_backtesting.csv")
    if main_output_path.exists():
        existing_df = pd.read_csv(main_output_path)
        existing_df = pd.concat([existing_df, results_df], ignore_index=True)
        existing_df.to_csv(main_output_path, index=False)
        logger.info(f"Appended to main backtesting CSV: {main_output_path}")
    else:
        results_df.to_csv(main_output_path, index=False)
        logger.info(f"Created main backtesting CSV: {main_output_path}")
    
    return results_df


if __name__ == "__main__":
    results = run_horizon_backtesting()
    if results is not None:
        print("\nSample output rows (first 10):")
        print(results.head(10).to_string())
        print(f"\nTotal rows: {len(results)}")
        print(f"\nHorizon years covered: {sorted(results['test_year'].unique())}")
        print(f"Horizons: {sorted(results['horizon_days'].unique())}")
