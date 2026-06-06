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
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from aac_adoption.config import RANDOM_STATE
from aac_adoption.features.feature_sets import model_feature_columns
from aac_adoption.models.split import make_time_split
from aac_adoption.models.train_advanced import prepare_catboost_frame, categorical_features_for
from aac_adoption.models.train_boosting import make_boosting_preprocessor


def tune_models(df: pd.DataFrame, n_trials: int = 20) -> dict[str, Any]:
    """Run Optuna studies to find best hyperparameters."""
    split = make_time_split(df, "classification_target", animal_subset="combined")
    train_df = split.train.sort_values("intake_datetime").reset_index(drop=True)
    feature_columns = model_feature_columns(train_df)
    cat_features = categorical_features_for(feature_columns)
    
    # We need regression targets too
    # make_time_split for regression returns same train split size, but targets are different
    reg_split = make_time_split(df, "regression_target_days", animal_subset="combined")
    reg_train_df = reg_split.train.sort_values("intake_datetime").reset_index(drop=True)

    cv = TimeSeriesSplit(n_splits=5)
    best_params: dict[str, Any] = {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 1. CatBoost Classification
    cat_X = prepare_catboost_frame(train_df, feature_columns)
    cat_y = train_df["classification_target"]
    
    def catboost_clf_objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": RANDOM_STATE,
            "verbose": False,
        }
        scores = []
        for train_idx, val_idx in cv.split(cat_X):
            X_tr, y_tr = cat_X.iloc[train_idx], cat_y.iloc[train_idx]
            X_va, y_va = cat_X.iloc[val_idx], cat_y.iloc[val_idx]
            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_va, y_va), early_stopping_rounds=20)
            preds = model.predict_proba(X_va)[:, 1]
            scores.append(roc_auc_score(y_va, preds))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(catboost_clf_objective, n_trials=n_trials)
    best_params["catboost_classification"] = study.best_params

    # 2. CatBoost Regression
    cat_y_reg = reg_train_df["regression_target_days"]
    
    def catboost_reg_objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "random_seed": RANDOM_STATE,
            "verbose": False,
        }
        scores = []
        for train_idx, val_idx in cv.split(cat_X):
            X_tr, y_tr = cat_X.iloc[train_idx], cat_y_reg.iloc[train_idx]
            X_va, y_va = cat_X.iloc[val_idx], cat_y_reg.iloc[val_idx]
            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_va, y_va), early_stopping_rounds=20)
            preds = model.predict(X_va)
            scores.append(mean_absolute_error(y_va, preds))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(catboost_reg_objective, n_trials=n_trials)
    best_params["catboost_regression"] = study.best_params

    # 3. HistGradientBoosting Classification
    hist_X = train_df[feature_columns]
    preprocessor = make_boosting_preprocessor(hist_X)
    hist_X_transformed = preprocessor.fit_transform(hist_X)
    
    def hist_clf_objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_iter": trial.suggest_int("max_iter", 50, 200),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "random_state": RANDOM_STATE,
        }
        scores = []
        for train_idx, val_idx in cv.split(hist_X_transformed):
            X_tr, y_tr = hist_X_transformed[train_idx], cat_y.iloc[train_idx]
            X_va, y_va = hist_X_transformed[val_idx], cat_y.iloc[val_idx]
            model = HistGradientBoostingClassifier(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_va)[:, 1]
            scores.append(roc_auc_score(y_va, preds))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(hist_clf_objective, n_trials=n_trials)
    best_params["hist_gradient_boosting_classification"] = study.best_params

    # 4. HistGradientBoosting Regression
    def hist_reg_objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_iter": trial.suggest_int("max_iter", 50, 200),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "random_state": RANDOM_STATE,
        }
        scores = []
        for train_idx, val_idx in cv.split(hist_X_transformed):
            X_tr, y_tr = hist_X_transformed[train_idx], cat_y_reg.iloc[train_idx]
            X_va, y_va = hist_X_transformed[val_idx], cat_y_reg.iloc[val_idx]
            model = HistGradientBoostingRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            scores.append(mean_absolute_error(y_va, preds))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(hist_reg_objective, n_trials=n_trials)
    best_params["hist_gradient_boosting_regression"] = study.best_params

    return best_params

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
