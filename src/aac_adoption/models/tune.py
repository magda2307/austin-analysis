"""Hyperparameter tuning for core models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, CatBoostError
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from aac_adoption.config import RANDOM_STATE
from aac_adoption.features.feature_sets import model_feature_columns
from aac_adoption.models.split import make_time_split
from aac_adoption.models.train_advanced import prepare_catboost_frame, categorical_features_for
from aac_adoption.models.train_boosting import make_boosting_preprocessor


def tune_models(
    df: pd.DataFrame,
    n_trials: int = 20,
    sampler_type: str = "tpe",
    max_iterations: int = 5000,
    cv_splits: int = 5,
) -> tuple[dict[str, Any], dict[str, optuna.Study]]:
    """Run Optuna studies to find best hyperparameters.
    
    Args:
        df: Training dataframe
        n_trials: Number of trials per model
        sampler_type: 'tpe', 'random', or 'cmaes'
    """
    # 1. Decoupled Classification Setup
    # Sort chronologically by intake_datetime before setting up TimeSeriesSplit
    split_clf = make_time_split(df, "classification_target", animal_subset="combined")
    train_df_clf = split_clf.train.sort_values("intake_datetime").reset_index(drop=True)
    feature_columns_clf = model_feature_columns(train_df_clf)
    cat_features_clf = categorical_features_for(feature_columns_clf)
    X_clf = train_df_clf[feature_columns_clf]
    y_clf = train_df_clf["classification_target"]
    cat_X_clf = prepare_catboost_frame(train_df_clf, feature_columns_clf)
    
    # 2. Decoupled Regression Setup
    # Sort chronologically by intake_datetime before setting up TimeSeriesSplit
    split_reg = make_time_split(df, "regression_target_days", animal_subset="combined")
    train_df_reg = split_reg.train.sort_values("intake_datetime").reset_index(drop=True)
    feature_columns_reg = model_feature_columns(train_df_reg)
    cat_features_reg = categorical_features_for(feature_columns_reg)
    X_reg = train_df_reg[feature_columns_reg]
    y_reg = train_df_reg["regression_target_days"]
    cat_X_reg = prepare_catboost_frame(train_df_reg, feature_columns_reg)

    # Chronological cross-validation using TimeSeriesSplit with 5 splits.
    # This prevents data leakage by ensuring that training folds always precede
    # validation folds chronologically, matching the temporal splitting strategy.
    # Since the input DataFrames are sorted chronologically by intake_datetime,
    # the splits created by TimeSeriesSplit represent successive chronological steps.
    cv = TimeSeriesSplit(n_splits=cv_splits)
    
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
    def catboost_clf_objective(trial: optuna.Trial) -> float:
        """Objective function for CatBoost classification hyperparameter tuning.
        
        Error handling strategy:
        - CatBoostError: Trial is pruned (invalid configuration)
        - ValueError/data issues: Trial is pruned (problem with data)
        - Unexpected errors: Bubble up with clear context
        """
        params = {
            "iterations": max_iterations,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "bootstrap_type": "Bernoulli",
            "loss_function": "Logloss",
            "eval_metric": "PRAUC",
            "random_seed": RANDOM_STATE,
            "verbose": False,
            "auto_class_weights": "Balanced",
        }
        
        try:
            scores = []
            for train_idx, val_idx in cv.split(cat_X_clf):
                X_tr, y_tr = cat_X_clf.iloc[train_idx], y_clf.iloc[train_idx]
                X_va, y_va = cat_X_clf.iloc[val_idx], y_clf.iloc[val_idx]
                
                model = CatBoostClassifier(**params)
                model.fit(
                    X_tr, y_tr,
                    cat_features=cat_features_clf,
                    eval_set=(X_va, y_va),
                    early_stopping_rounds=50,
                )
                preds = model.predict_proba(X_va)[:, 1]
                scores.append(average_precision_score(y_va, preds))
            return np.mean(scores)
        except CatBoostError as e:
            raise optuna.TrialPruned(f"CatBoost failed: {e}")
        except ValueError as e:
            raise optuna.TrialPruned(f"Invalid data/configuration: {e}")

    study_cat_clf = optuna.create_study(direction="maximize", sampler=get_sampler(), pruner=optuna.pruners.MedianPruner())
    study_cat_clf.optimize(catboost_clf_objective, n_trials=n_trials)
    best_params["catboost_classification"] = _get_study_result(study_cat_clf)

    # 2. CatBoost Regression
    def catboost_reg_objective(trial: optuna.Trial) -> float:
        """Objective function for CatBoost regression hyperparameter tuning.
        
        Error handling strategy:
        - CatBoostError: Trial is pruned (invalid configuration)
        - ValueError/data issues: Trial is pruned (problem with data)
        - Unexpected errors: Bubble up with clear context
        """
        params = {
            "iterations": max_iterations,
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
        
        try:
            scores = []
            for train_idx, val_idx in cv.split(cat_X_reg):
                X_tr, y_tr_reg = cat_X_reg.iloc[train_idx], y_reg.iloc[train_idx]
                X_va, y_va_reg = cat_X_reg.iloc[val_idx], y_reg.iloc[val_idx]

                model = CatBoostRegressor(**params)
                model.fit(
                    X_tr, np.log1p(y_tr_reg),
                    cat_features=cat_features_reg,
                    eval_set=(X_va, np.log1p(y_va_reg)),
                    early_stopping_rounds=50,
                )
                preds_log = model.predict(X_va)
                preds = np.expm1(preds_log)
                scores.append(mean_absolute_error(y_va_reg, preds))
            return np.mean(scores)
        except CatBoostError as e:
            raise optuna.TrialPruned(f"CatBoost failed: {e}")
        except ValueError as e:
            raise optuna.TrialPruned(f"Invalid data/configuration: {e}")

    study_cat_reg = optuna.create_study(direction="minimize", sampler=get_sampler(), pruner=optuna.pruners.MedianPruner())
    study_cat_reg.optimize(catboost_reg_objective, n_trials=n_trials)
    best_params["catboost_regression"] = _get_study_result(study_cat_reg)

    # 3. HistGradientBoosting Classification
    def hist_clf_objective(trial: optuna.Trial) -> float:
        """Objective function for HistGradientBoosting classification hyperparameter tuning.
        
        Error handling strategy:
        - ValueError/data issues: Trial is pruned (problem with data)
        - Unexpected errors: Bubble up with clear context
        
        NOTE: Broad except blocks turn tuning failures into valid trials.
        Only specific exceptions should be caught and pruned.
        """
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_iter": max_iterations,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 50,
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
        }
        
        try:
            scores = []
            for train_idx, val_idx in cv.split(X_clf):
                X_tr, y_tr = X_clf.iloc[train_idx], y_clf.iloc[train_idx]
                X_va, y_va = X_clf.iloc[val_idx], y_clf.iloc[val_idx]
                
                # Preprocessor fitted strictly inside CV loop on training subset
                preprocessor = make_boosting_preprocessor(X_tr)
                X_tr_transformed = preprocessor.fit_transform(X_tr)
                X_va_transformed = preprocessor.transform(X_va)

                # Disable early stopping if fold has fewer than 100 samples to avoid split crash
                fold_params = {**params, "early_stopping": (len(X_tr) >= 100)}
                model = HistGradientBoostingClassifier(**fold_params)
                model.fit(X_tr_transformed, y_tr)
                preds = model.predict_proba(X_va_transformed)[:, 1]
                scores.append(average_precision_score(y_va, preds))
            return np.mean(scores)
        except ValueError as e:
            raise optuna.TrialPruned(f"Invalid data/configuration: {e}")

    study_hist_clf = optuna.create_study(direction="maximize", sampler=get_sampler())
    study_hist_clf.optimize(hist_clf_objective, n_trials=n_trials)
    best_params["hist_gradient_boosting_classification"] = _get_study_result(study_hist_clf)

    # 4. HistGradientBoosting Regression
    def hist_reg_objective(trial: optuna.Trial) -> float:
        """Objective function for HistGradientBoosting regression hyperparameter tuning.
        
        Error handling strategy:
        - ValueError/data issues: Trial is pruned (problem with data)
        - Unexpected errors: Bubble up with clear context
        
        NOTE: Broad except blocks turn tuning failures into valid trials.
        Only specific exceptions should be caught and pruned.
        """
        params = {
            "loss": "absolute_error",
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_iter": max_iterations,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 50,
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
            "random_state": RANDOM_STATE,
        }
        
        try:
            scores = []
            for train_idx, val_idx in cv.split(X_reg):
                X_tr, y_tr_reg = X_reg.iloc[train_idx], y_reg.iloc[train_idx]
                X_va, y_va_reg = X_reg.iloc[val_idx], y_reg.iloc[val_idx]
                
                # Preprocessor fitted strictly inside CV loop on training subset
                preprocessor = make_boosting_preprocessor(X_tr)
                X_tr_transformed = preprocessor.fit_transform(X_tr)
                X_va_transformed = preprocessor.transform(X_va)

                # Disable early stopping if fold has fewer than 100 samples to avoid split crash
                fold_params = {**params, "early_stopping": (len(X_tr) >= 100)}
                model = HistGradientBoostingRegressor(**fold_params)
                model.fit(X_tr_transformed, y_tr_reg)
                preds = model.predict(X_va_transformed)
                scores.append(mean_absolute_error(y_va_reg, preds))
            return np.mean(scores)
        except ValueError as e:
            raise optuna.TrialPruned(f"Invalid data/configuration: {e}")

    study_hist_reg = optuna.create_study(direction="minimize", sampler=get_sampler())
    study_hist_reg.optimize(hist_reg_objective, n_trials=n_trials)
    best_params["hist_gradient_boosting_regression"] = _get_study_result(study_hist_reg)

    studies = {
        "catboost_classification": study_cat_clf,
        "catboost_regression": study_cat_reg,
        "hist_gradient_boosting_classification": study_hist_clf,
        "hist_gradient_boosting_regression": study_hist_reg,
    }

    return best_params, studies

def _get_study_result(study: optuna.Study) -> dict[str, Any]:
    try:
        best_params = study.best_params
        best_value = study.best_value
        status = "ok"
        failure_reason = None
    except ValueError as e:
        best_params = None
        best_value = None
        status = "failed"
        failure_reason = str(e)

    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    return {
        "status": status,
        "best_params": best_params,
        "best_value": best_value,
        "completed_trials": completed_trials,
        "pruned_trials": pruned_trials,
        "failure_reason": failure_reason,
    }

def run_tuning(df: pd.DataFrame, output_dir: Path, n_trials: int = 20):
    """Run full tuning suite and save results."""
    best_params, studies = tune_models(df, n_trials=n_trials)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=2)
        
    for name, study in studies.items():
        study.trials_dataframe().to_csv(output_dir / f"{name}_trials.csv", index=False)

def save_tuned_params(params: dict[str, Any], output_path: Path) -> None:
    """Save best hyperparameters to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)
