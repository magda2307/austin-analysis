"""Business logic for comparing recency strategies in model training."""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, brier_score_loss
from catboost import CatBoostClassifier, CatBoostRegressor

from aac_adoption.features.feature_sets import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMNS, METADATA_COLUMNS
from aac_adoption.models.evaluate import expected_calibration_error
from aac_adoption.models.bootstrap import bootstrap_ci

logger = logging.getLogger(__name__)


def compute_recency_weights(df_train: pd.DataFrame, start_year: int = 2013, end_year: int = 2021) -> pd.Series:
    """Compute sample weights dynamically based on intake_year."""
    span = end_year - start_year
    if span <= 0:
        return pd.Series(1.0, index=df_train.index)
    weights = df_train["intake_year"].apply(
        lambda y: 1.0 + 0.5 * (y - start_year) / span if pd.notnull(y) else 1.0
    )
    return weights.clip(lower=1.0, upper=1.5)


def run_recency_comparison(
    df: pd.DataFrame,
    n_bootstraps: int = 100,
    iterations: int = 300,
    test_period: str = "2024-2025",
    quick: bool = False,
) -> pd.DataFrame:
    """Evaluate and compare training recency strategies on a test period.

    Strategies compared:
      - full_history: training on full history (2013 to train_end)
      - recent_5yr: training on most recent 5 years of training data
      - recent_3yr: training on most recent 3 years of training data
      - recency_weighted: training on full history with linear recency weights

    Args:
        df: Processed modeling dataset
        n_bootstraps: Number of bootstrap iterations for CI calculation
        iterations: CatBoost model iterations (trees)
        test_period: Test period years, e.g. "2024-2025"
        quick: Quick mode override (uses lower default iterations/bootstraps)

    Returns:
        DataFrame with strategy evaluation results
    """
    if quick:
        n_bootstraps = n_bootstraps if n_bootstraps != 100 else 5
        iterations = iterations if iterations != 300 else 20

    logger.info(f"Starting recency comparison with n_bootstraps={n_bootstraps}, iterations={iterations}, test_period={test_period}")

    # Parse test period
    try:
        parts = test_period.split("-")
        test_years = list(range(int(parts[0]), int(parts[1]) + 1))
    except Exception:
        test_years = [2024, 2025]
    
    test_start = min(test_years)
    train_end = test_start - 3  # Leave a 2-year validation gap before the test period (e.g., 2021 for 2024 test start)

    # Dynamic training year ranges
    strategies = [
        {
            "name": "full_history",
            "train_years": list(range(2013, train_end + 1)),
            "weighted": False
        },
        {
            "name": "recent_5yr",
            "train_years": list(range(max(2013, train_end - 4), train_end + 1)),
            "weighted": False
        },
        {
            "name": "recent_3yr",
            "train_years": list(range(max(2013, train_end - 2), train_end + 1)),
            "weighted": False
        },
        {
            "name": "recency_weighted",
            "train_years": list(range(2013, train_end + 1)),
            "weighted": True
        },
    ]

    # Preprocess features (avoiding leakage columns)
    feature_cols = list(dict.fromkeys([c for c in (CATEGORICAL_FEATURES + NUMERIC_FEATURES) if c in df.columns]))
    feature_cols = [c for c in feature_cols if c not in TARGET_COLUMNS and c not in METADATA_COLUMNS]

    X = df[feature_cols].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown").astype(str)
    for col in NUMERIC_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna(0)

    y_cls = df["classification_target"]
    y_reg = df["regression_target_days"]
    animal_ids = df["animal_id"] if "animal_id" in df.columns else None

    # Build evaluation mask
    test_mask = df["intake_year"].isin(test_years)
    X_test = X[test_mask]
    y_test_c = y_cls[test_mask]
    y_test_r = y_reg[test_mask]

    cat_features = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    results = []

    # 1. Evaluate on all/combined subset
    for strat in strategies:
        logger.info(f"Evaluating strategy: {strat['name']} on combined subset")
        train_mask = df["intake_year"].isin(strat["train_years"])
        
        X_train = X[train_mask]
        y_train_c = y_cls[train_mask]
        y_train_r = y_reg[train_mask]
        
        if len(X_train) < 2 or len(X_test) < 2:
            continue

        sample_weight = None
        if strat["weighted"]:
            sample_weight = compute_recency_weights(df[train_mask], start_year=2013, end_year=train_end)

        # Train Classifier
        cb_cls = CatBoostClassifier(
            iterations=iterations,
            auto_class_weights="Balanced",
            cat_features=cat_features,
            verbose=0,
            random_seed=42
        )
        cb_cls.fit(X_train, y_train_c, sample_weight=sample_weight)
        preds_cb_c = cb_cls.predict_proba(X_test)[:, 1]
        
        roc_cb = roc_auc_score(y_test_c, preds_cb_c)
        pr_cb = average_precision_score(y_test_c, preds_cb_c)
        brier_cb = brier_score_loss(y_test_c, preds_cb_c)
        ece_cb = expected_calibration_error(y_test_c, preds_cb_c)
        
        ci_roc = bootstrap_ci(
            y_test_c.values,
            preds_cb_c,
            roc_auc_score,
            y_score=preds_cb_c,
            n_bootstraps=n_bootstraps,
            animal_ids=animal_ids[test_mask].values if animal_ids is not None else None
        )
        ci_pr = bootstrap_ci(
            y_test_c.values,
            preds_cb_c,
            average_precision_score,
            y_score=preds_cb_c,
            n_bootstraps=n_bootstraps,
            animal_ids=animal_ids[test_mask].values if animal_ids is not None else None
        )
        
        # Train Regressor
        cb_reg = CatBoostRegressor(
            iterations=iterations,
            cat_features=cat_features,
            verbose=0,
            random_seed=42
        )
        cb_reg.fit(X_train, y_train_r, sample_weight=sample_weight)
        preds_cb_r = cb_reg.predict(X_test)
        mae_cb = mean_absolute_error(y_test_r, preds_cb_r)
        
        ci_mae = bootstrap_ci(
            y_test_r.values,
            preds_cb_r,
            mean_absolute_error,
            n_bootstraps=n_bootstraps,
            animal_ids=animal_ids[test_mask].values if animal_ids is not None else None
        )

        train_years_str = f"{min(strat['train_years'])}-{max(strat['train_years'])}"
        results.append({
            "strategy": strat["name"],
            "train_years": train_years_str,
            "test_years": test_period,
            "subset": "combined",
            "model": "CatBoost",
            "pr_auc": float(pr_cb),
            "pr_auc_lower": float(ci_pr[0]),
            "pr_auc_upper": float(ci_pr[1]),
            "roc_auc": float(roc_cb),
            "roc_auc_lower": float(ci_roc[0]),
            "roc_auc_upper": float(ci_roc[1]),
            "brier": float(brier_cb),
            "ece": float(ece_cb),
            "mae": float(mae_cb),
            "mae_lower": float(ci_mae[0]),
            "mae_upper": float(ci_mae[1]),
        })

    # 2. Evaluate on subgroups (dogs, cats)
    subgroups = [
        ("dogs", df["animal_type"].astype(str).str.lower().eq("dog")),
        ("cats", df["animal_type"].astype(str).str.lower().eq("cat"))
    ]
    for subset_name, subset_mask in subgroups:
        subset_df = df[subset_mask].copy()
        if subset_df.empty or len(subset_df) < 10:
            continue
        
        X_sub = subset_df[feature_cols].copy()
        for col in CATEGORICAL_FEATURES:
            if col in X_sub.columns:
                X_sub[col] = X_sub[col].fillna("Unknown").astype(str)
        for col in NUMERIC_FEATURES:
            if col in X_sub.columns:
                X_sub[col] = X_sub[col].fillna(0)
        
        y_sub_c = subset_df["classification_target"]
        y_sub_r = subset_df["regression_target_days"]
        animal_ids_sub = subset_df["animal_id"] if "animal_id" in subset_df.columns else None
        
        test_mask_sub = subset_df["intake_year"].isin(test_years)
        X_test_sub = X_sub[test_mask_sub]
        y_test_c_sub = y_sub_c[test_mask_sub]
        y_test_r_sub = y_sub_r[test_mask_sub]
        
        if len(X_test_sub) < 2 or len(np.unique(y_test_c_sub)) < 2:
            continue
        
        for strat in strategies:
            # Skip weighted for subgroups to align with original script design
            if strat["weighted"]:
                continue
            
            logger.info(f"Evaluating strategy: {strat['name']} on subset: {subset_name}")
            train_mask_sub = subset_df["intake_year"].isin(strat["train_years"])
            X_train_sub = X_sub[train_mask_sub]
            y_train_c_sub = y_sub_c[train_mask_sub]
            y_train_r_sub = y_sub_r[train_mask_sub]
            
            if len(X_train_sub) < 2:
                continue
            
            cb_cls_sub = CatBoostClassifier(
                iterations=iterations,
                auto_class_weights="Balanced",
                cat_features=cat_features,
                verbose=0,
                random_seed=42
            )
            cb_cls_sub.fit(X_train_sub, y_train_c_sub)
            preds_cb_c_sub = cb_cls_sub.predict_proba(X_test_sub)[:, 1]
            
            roc_cb_sub = roc_auc_score(y_test_c_sub, preds_cb_c_sub)
            pr_cb_sub = average_precision_score(y_test_c_sub, preds_cb_c_sub)
            brier_cb_sub = brier_score_loss(y_test_c_sub, preds_cb_c_sub)
            ece_cb_sub = expected_calibration_error(y_test_c_sub, preds_cb_c_sub)
            
            ci_roc_sub = bootstrap_ci(
                y_test_c_sub.values,
                preds_cb_c_sub,
                roc_auc_score,
                y_score=preds_cb_c_sub,
                n_bootstraps=n_bootstraps,
                animal_ids=animal_ids_sub[test_mask_sub].values if animal_ids_sub is not None else None
            )
            ci_pr_sub = bootstrap_ci(
                y_test_c_sub.values,
                preds_cb_c_sub,
                average_precision_score,
                y_score=preds_cb_c_sub,
                n_bootstraps=n_bootstraps,
                animal_ids=animal_ids_sub[test_mask_sub].values if animal_ids_sub is not None else None
            )
            
            cb_reg_sub = CatBoostRegressor(
                iterations=iterations,
                cat_features=cat_features,
                verbose=0,
                random_seed=42
            )
            cb_reg_sub.fit(X_train_sub, y_train_r_sub)
            preds_cb_r_sub = cb_reg_sub.predict(X_test_sub)
            mae_cb_sub = mean_absolute_error(y_test_r_sub, preds_cb_r_sub)
            
            ci_mae_sub = bootstrap_ci(
                y_test_r_sub.values,
                preds_cb_r_sub,
                mean_absolute_error,
                n_bootstraps=n_bootstraps,
                animal_ids=animal_ids_sub[test_mask_sub].values if animal_ids_sub is not None else None
            )
            
            train_years_str = f"{min(strat['train_years'])}-{max(strat['train_years'])}"
            results.append({
                "strategy": strat["name"],
                "train_years": train_years_str,
                "test_years": test_period,
                "subset": subset_name,
                "model": "CatBoost",
                "pr_auc": float(pr_cb_sub),
                "pr_auc_lower": float(ci_pr_sub[0]),
                "pr_auc_upper": float(ci_pr_sub[1]),
                "roc_auc": float(roc_cb_sub),
                "roc_auc_lower": float(ci_roc_sub[0]),
                "roc_auc_upper": float(ci_roc_sub[1]),
                "brier": float(brier_cb_sub),
                "ece": float(ece_cb_sub),
                "mae": float(mae_cb_sub),
                "mae_lower": float(ci_mae_sub[0]),
                "mae_upper": float(ci_mae_sub[1]),
            })

    return pd.DataFrame(results)


def plot_performance_comparison(df: pd.DataFrame, path: Path) -> None:
    """Create performance comparison visualization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metric_titles = [
        ("pr_auc", "PR-AUC (Primary Classification)"),
        ("roc_auc", "ROC-AUC"),
        ("brier", "Brier Score"),
        ("ece", "ECE"),
        ("mae", "MAE (Regression)"),
    ]
    
    subsets = sorted(df["subset"].unique())
    strategies = sorted(df["strategy"].unique())
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    
    for idx, (metric_col, metric_title) in enumerate(metric_titles):
        if metric_col not in df.columns:
            continue
        ax = axes[idx]
        
        x_pos = np.arange(len(subsets))
        width = 0.8 / len(strategies)
        
        for i, strategy in enumerate(strategies):
            means = []
            yerr_lower = []
            yerr_upper = []
            
            for s in subsets:
                subset_strat_df = df[(df["subset"] == s) & (df["strategy"] == strategy)]
                if len(subset_strat_df) > 0:
                    val = subset_strat_df[metric_col].values[0]
                    means.append(val)
                    
                    lower_col = f"{metric_col}_lower"
                    upper_col = f"{metric_col}_upper"
                    if lower_col in df.columns and upper_col in df.columns:
                        l = subset_strat_df[lower_col].values[0]
                        u = subset_strat_df[upper_col].values[0]
                        if not pd.isna(l) and not pd.isna(u):
                            yerr_lower.append(val - l)
                            yerr_upper.append(u - val)
                        else:
                            yerr_lower.append(0.0)
                            yerr_upper.append(0.0)
                    else:
                        yerr_lower.append(0.0)
                        yerr_upper.append(0.0)
                else:
                    means.append(np.nan)
                    yerr_lower.append(0.0)
                    yerr_upper.append(0.0)
            
            yerr = [yerr_lower, yerr_upper]
            rects = ax.bar(x_pos + i * width, means, width, label=strategy, color=colors[i])
            # Add error bars
            ax.errorbar(x_pos + i * width, means, yerr=yerr, fmt='none', ecolor='black', capsize=3, alpha=0.6)
        
        ax.set_xlabel("Animal Subset")
        ax.set_ylabel(metric_title)
        ax.set_title(f"Model Performance: {metric_title}")
        ax.set_xticks(x_pos + width * (len(strategies) - 1) / 2)
        ax.set_xticklabels(subsets)
        ax.legend(loc="upper right" if idx != 4 else "lower right")
        ax.grid(axis="y", alpha=0.3)
    
    axes[-1].axis("off")
    
    plt.suptitle("Recency Strategy Comparison - All Subsets (2024-2025 Test Period)", 
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Performance comparison plot saved to {path}")
