"""Hyperparameter tuning for core models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from aac_adoption.config import RANDOM_STATE
from aac_adoption.features.feature_sets import model_feature_columns
from aac_adoption.models.split import make_time_split
from aac_adoption.models.train_advanced import prepare_catboost_frame, categorical_features_for
from aac_adoption.models.train_boosting import make_boosting_preprocessor

try:
    from optuna.integration import CatBoostPruningCallback
except ImportError:
    CatBoostPruningCallback = None


def tune_models(df: pd.DataFrame, n_trials: int = 20, sampler_type: str = "tpe") -> tuple[dict[str, Any], dict[str, optuna.Study]]:
    """Run Optuna studies to find best hyperparameters.
    
    Args:
        df: Training dataframe
        n_trials: Number of trials per model
        sampler_type: 'tpe', 'random', or 'cmaes'
    """
    split = make_time_split(df, "classification_target", animal_subset="combined")
    train_df = split.train.sort_values("intake_datetime").reset_index(drop=True)
    feature_columns = model_feature_columns(train_df)
    cat_features = categorical_features_for(feature_columns)
    
    reg_split = make_time_split(df, "regression_target_days", animal_subset="combined")
    reg_train_df = reg_split.train.sort_values("intake_datetime").reset_index(drop=True)

    # Use only the last chronological split for tuning to save time & support deep pruning
    cv = TimeSeriesSplit(n_splits=5)
    
    best_params: dict[str, Any] = {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def get_sampler():
        if sampler_type == "tpe":
            return optuna.samplers.TPESampler(seed=RANDOM_STATE)
        elif sampler_type == "random":
            return optuna.samplers.RandomSampler(seed=RANDOM_STATE)
        elif sampler_type == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=RANDOM_STATE)
        else:
            raise ValueError(f"Unsupported sampler_type: {sampler_type}")

    # 1. CatBoost Classification
    cat_X = prepare_catboost_frame(train_df, feature_columns)
    cat_y = train_df["classification_target"]
    
    splits = list(cv.split(cat_X))
    train_idx, val_idx = splits[-1]
    
    X_tr_cat, y_tr_cat = cat_X.iloc[train_idx], cat_y.iloc[train_idx]
    X_va_cat, y_va_cat = cat_X.iloc[val_idx], cat_y.iloc[val_idx]
    
    def catboost_clf_objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": 5000,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "bootstrap_type": "Bernoulli",
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": RANDOM_STATE,
            "verbose": False,
            "auto_class_weights": "Balanced",
        }
        
        callbacks = []
        if CatBoostPruningCallback is not None:
            callbacks.append(CatBoostPruningCallback(trial, "AUC"))
            
        model = CatBoostClassifier(**params)
        model.fit(
            X_tr_cat, y_tr_cat,
            cat_features=cat_features,
            eval_set=(X_va_cat, y_va_cat),
            early_stopping_rounds=50,
            callbacks=callbacks
        )
        
        preds = model.predict_proba(X_va_cat)[:, 1]
        return average_precision_score(y_va_cat, preds)

    study_cat_clf = optuna.create_study(direction="maximize", sampler=get_sampler(), pruner=optuna.pruners.MedianPruner())
    study_cat_clf.optimize(catboost_clf_objective, n_trials=n_trials)
    best_params["catboost_classification"] = study_cat_clf.best_params

    # 2. CatBoost Regression
    cat_y_reg = reg_train_df["regression_target_days"]
    y_tr_cat_reg, y_va_cat_reg = cat_y_reg.iloc[train_idx], cat_y_reg.iloc[val_idx]
    
    def catboost_reg_objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": 5000,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "bootstrap_type": "Bernoulli",
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "random_seed": RANDOM_STATE,
            "verbose": False,
        }
        
        callbacks = []
        if CatBoostPruningCallback is not None:
            callbacks.append(CatBoostPruningCallback(trial, "MAE"))
            
        model = CatBoostRegressor(**params)
        model.fit(
            X_tr_cat, y_tr_cat_reg,
            cat_features=cat_features,
            eval_set=(X_va_cat, y_va_cat_reg),
            early_stopping_rounds=50,
            callbacks=callbacks
        )
        preds = model.predict(X_va_cat)
        return mean_absolute_error(y_va_cat_reg, preds)

    study_cat_reg = optuna.create_study(direction="minimize", sampler=get_sampler(), pruner=optuna.pruners.MedianPruner())
    study_cat_reg.optimize(catboost_reg_objective, n_trials=n_trials)
    best_params["catboost_regression"] = study_cat_reg.best_params

    # 3. HistGradientBoosting Classification
    hist_X = train_df[feature_columns]
    preprocessor = make_boosting_preprocessor(hist_X)
    hist_X_transformed = preprocessor.fit_transform(hist_X)
    
    X_tr_hist, y_tr_hist = hist_X_transformed[train_idx], cat_y.iloc[train_idx]
    X_va_hist, y_va_hist = hist_X_transformed[val_idx], cat_y.iloc[val_idx]
    
    def hist_clf_objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_iter": 5000,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 50,
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
        }
        model = HistGradientBoostingClassifier(**params)
        model.fit(X_tr_hist, y_tr_hist)
        preds = model.predict_proba(X_va_hist)[:, 1]
        return average_precision_score(y_va_hist, preds)

    study_hist_clf = optuna.create_study(direction="maximize", sampler=get_sampler())
    study_hist_clf.optimize(hist_clf_objective, n_trials=n_trials)
    best_params["hist_gradient_boosting_classification"] = study_hist_clf.best_params

    # 4. HistGradientBoosting Regression
    y_tr_hist_reg, y_va_hist_reg = cat_y_reg.iloc[train_idx], cat_y_reg.iloc[val_idx]
    
    def hist_reg_objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_iter": 5000,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 50,
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
            "random_state": RANDOM_STATE,
        }
        model = HistGradientBoostingRegressor(**params)
        model.fit(X_tr_hist, y_tr_hist_reg)
        preds = model.predict(X_va_hist)
        return mean_absolute_error(y_va_hist_reg, preds)

    study_hist_reg = optuna.create_study(direction="minimize", sampler=get_sampler())
    study_hist_reg.optimize(hist_reg_objective, n_trials=n_trials)
    best_params["hist_gradient_boosting_regression"] = study_hist_reg.best_params

    studies = {
        "catboost_classification": study_cat_clf,
        "catboost_regression": study_cat_reg,
        "hist_gradient_boosting_classification": study_hist_clf,
        "hist_gradient_boosting_regression": study_hist_reg,
    }

    return best_params, studies

def save_tuned_params(params: dict[str, Any], path: Path):
    """Save best parameters to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2), encoding="utf-8")

def load_tuned_params(path: Path | str) -> dict[str, Any] | None:
    """Load tuned params if they exist."""
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None
