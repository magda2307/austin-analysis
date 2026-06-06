"""Compare recency strategies for model training."""

import pandas as pd
from pathlib import Path
import logging
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, brier_score_loss
from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
from aac_adoption.features.feature_sets import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from aac_adoption.models.evaluate import expected_calibration_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_recency_weights(df_train):
    """Compute sample weights dynamically based on intake_year."""
    weights = df_train["intake_year"].apply(
        lambda y: 1.0 + 0.5 * (y - 2013) / (2021 - 2013) if pd.notnull(y) else 1.0
    )
    return weights.clip(lower=1.0, upper=1.5)

def main():
    logger.info("Loading dataset for recency comparison...")
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

    # Final test period is 2024-2025
    test_mask = df["intake_year"].isin([2024, 2025])
    X_test, y_test_c, y_test_r = X[test_mask], y_cls[test_mask], y_reg[test_mask]

    cat_features = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    
    strategies = [
        {"name": "full_history_unweighted", "train_years": list(range(2013, 2022)), "weighted": False},
        {"name": "recent_5_year", "train_years": list(range(2017, 2022)), "weighted": False},
        {"name": "recent_3_year", "train_years": list(range(2019, 2022)), "weighted": False},
        {"name": "recency_weighted", "train_years": list(range(2013, 2022)), "weighted": True},
    ]

    results = []

    for strat in strategies:
        logger.info(f"Evaluating strategy: {strat['name']}")
        train_mask = df["intake_year"].isin(strat["train_years"])
        
        X_train = X[train_mask]
        y_train_c = y_cls[train_mask]
        y_train_r = y_reg[train_mask]
        
        sample_weight = None
        if strat["weighted"]:
            sample_weight = compute_recency_weights(df[train_mask])

        if len(X_train) == 0:
            continue

        # 1. CatBoost Classification
        logger.info("  Training CatBoost Classifier...")
        cb_cls = CatBoostClassifier(iterations=300, auto_class_weights="Balanced", cat_features=cat_features, verbose=0, random_seed=42)
        cb_cls.fit(X_train, y_train_c, sample_weight=sample_weight)
        preds_cb_c = cb_cls.predict_proba(X_test)[:, 1]
        
        roc_cb = roc_auc_score(y_test_c, preds_cb_c)
        pr_cb = average_precision_score(y_test_c, preds_cb_c)
        brier_cb = brier_score_loss(y_test_c, preds_cb_c)
        ece_cb = expected_calibration_error(y_test_c, preds_cb_c)

        # 2. CatBoost Regression
        logger.info("  Training CatBoost Regressor...")
        cb_reg = CatBoostRegressor(iterations=300, cat_features=cat_features, verbose=0, random_seed=42)
        cb_reg.fit(X_train, y_train_r, sample_weight=sample_weight)
        preds_cb_r = cb_reg.predict(X_test)
        mae_cb = mean_absolute_error(y_test_r, preds_cb_r)

        results.append({
            "strategy": strat["name"],
            "train_years": f"{min(strat['train_years'])}-{max(strat['train_years'])}",
            "test_years": "2024-2025",
            "subset": "all",
            "model": "CatBoost",
            "pr_auc": float(pr_cb),
            "roc_auc": float(roc_cb),
            "brier": float(brier_cb),
            "ece": float(ece_cb),
            "mae": float(mae_cb),
        })

    out_df = pd.DataFrame(results)
    out_path = Path("reports/tables/recency_strategy_comparison.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info(f"Saved recency strategy comparison to {out_path}")

if __name__ == "__main__":
    main()
