"""Evaluate model performance on rolling yearly windows to detect temporal drift."""

import pandas as pd
from pathlib import Path
import logging
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, brier_score_loss
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import numpy as np
from aac_adoption.models.train_advanced import prepare_catboost_frame
from aac_adoption.features.feature_sets import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_train_test_years(df):
    years = sorted(df["intake_year"].unique())
    splits = []
    # Train up to X, test X+1
    for i in range(len(years) - 1):
        if years[i+1] < 2019:
            continue
        train_years = years[:i+1]
        test_year = years[i+1]
        splits.append((train_years, test_year))
    return splits

def main():
    logger.info("Loading dataset for yearly backtesting...")
    data_path = Path("data/processed/modeling_dataset.csv")
    if not data_path.exists():
        logger.error(f"Missing {data_path}")
        return

    df = pd.read_csv(data_path)
    # Define features
    feature_cols = list(dict.fromkeys([c for c in (CATEGORICAL_FEATURES + NUMERIC_FEATURES) if c in df.columns]))
    X = df[feature_cols].copy()
    
    # Fill NAs
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown").astype(str)
    for col in NUMERIC_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna(0)

    y_cls = df["classification_target"]
    y_reg = df["regression_target_days"]

    splits = get_train_test_years(df)
    results = []

    cat_features = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    for train_years, test_year in splits:
        logger.info(f"Training on {min(train_years)}-{max(train_years)}, Testing on {test_year}")
        train_mask = df["intake_year"].isin(train_years)
        test_mask = df["intake_year"] == test_year

        X_train, X_test = X[train_mask], X[test_mask]
        y_train_c, y_test_c = y_cls[train_mask], y_cls[test_mask]
        y_train_r, y_test_r = y_reg[train_mask], y_reg[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        # 1. CatBoost Classification
        logger.info("  Training CatBoost Classifier...")
        cb_cls = CatBoostClassifier(iterations=300, auto_class_weights="Balanced", cat_features=cat_features, verbose=0, random_seed=42)
        cb_cls.fit(X_train, y_train_c)
        preds_cb_c = cb_cls.predict_proba(X_test)[:, 1]
        roc_cb = roc_auc_score(y_test_c, preds_cb_c)
        pr_cb = average_precision_score(y_test_c, preds_cb_c)

        # 2. CatBoost Regression
        logger.info("  Training CatBoost Regressor...")
        cb_reg = CatBoostRegressor(iterations=300, cat_features=cat_features, verbose=0, random_seed=42)
        cb_reg.fit(X_train, y_train_r)
        preds_cb_r = cb_reg.predict(X_test)
        mae_cb = mean_absolute_error(y_test_r, preds_cb_r)
        
        brier_cb = brier_score_loss(y_test_c, preds_cb_c)
        from aac_adoption.models.evaluate import expected_calibration_error
        ece_cb = expected_calibration_error(y_test_c, preds_cb_c)

        results.append({
            "train_years": f"{min(train_years)}-{max(train_years)}",
            "test_year": test_year,
            "subset": "all",
            "model": "CatBoost",
            "pr_auc": float(pr_cb),
            "roc_auc": float(roc_cb),
            "brier": float(brier_cb),
            "ece": float(ece_cb),
            "mae": float(mae_cb),
        })

    out_df = pd.DataFrame(results)
    out_path = Path("reports/tables/yearly_backtesting.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info(f"Saved yearly backtesting to {out_path}")

if __name__ == "__main__":
    main()
