\
"""Ensemble methods for combining multiple models."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, validate_data
from sklearn.model_selection import StratifiedKFold, KFold
from aac_adoption.config import RANDOM_STATE


def _safe_slice(arr, indices):
    if hasattr(arr, "iloc"):
        return arr.iloc[indices]
    if isinstance(arr, np.ndarray):
        return arr[indices]
    return np.asarray(arr)[indices]


class WeightedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Weighted ensemble of classifiers with calibrated probabilities."""
    
    def __init__(self, estimators: list, weights: list[float] = None):
        self.estimators = estimators
        self.weights = weights
    
    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)
        if self.weights is None:
            self.weights_ = [1.0 / len(self.estimators)] * len(self.estimators)
        else:
            self.weights_ = self.weights
        self.estimators_ = []
        for est in self.estimators:
            self.estimators_.append(clone(est).fit(X, y))
        return self
    
    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        
        all_probs = np.array([est.predict_proba(X) for est in self.estimators_])
        weighted_probs = np.average(all_probs, axis=0, weights=self.weights_)
        return weighted_probs
    
    def predict(self, X):
        X = validate_data(self, X, reset=False)
        proba = self.predict_proba(X)
        indices = (proba[:, 1] >= 0.5).astype(int)
        return self.classes_[indices]


class WeightedEnsembleRegressor(BaseEstimator, RegressorMixin):
    """Weighted ensemble of regressors."""
    
    def __init__(self, estimators: list, weights: list[float] = None):
        self.estimators = estimators
        self.weights = weights
    
    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        if self.weights is None:
            self.weights_ = [1.0 / len(self.estimators)] * len(self.estimators)
        else:
            self.weights_ = self.weights
            
        self.estimators_ = []
        for est in self.estimators:
            self.estimators_.append(clone(est).fit(X, y))
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        
        all_preds = np.array([est.predict(X) for est in self.estimators_])
        weighted_preds = np.average(all_preds, axis=0, weights=self.weights_)
        return weighted_preds


class StackedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Stacked ensemble with meta-learner."""
    
    def __init__(self, base_estimators: list, meta_estimator: Any, n_splits: int = 5, random_state: int = RANDOM_STATE):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.n_splits = n_splits
        self.random_state = random_state
    
    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)
        actual_n_splits = min(self.n_splits, len(X))
        unique_classes, class_counts = np.unique(y, return_counts=True)
        stratification_possible = all(count >= actual_n_splits for count in class_counts)
        
        if actual_n_splits < 2 or not stratification_possible:
            raise ValueError(
                f"Stacking ensemble requires at least {actual_n_splits}-fold cross-validation. "
                f"Got {len(X)} samples with {len(np.unique(y))} classes. "
                "Increase dataset size or use simple ensemble instead."
            )
        else:
            cv = StratifiedKFold(n_splits=actual_n_splits, shuffle=True, random_state=self.random_state)
            oof_predictions = np.zeros((len(X), len(self.base_estimators)))
            
            for train_idx, val_idx in cv.split(X, y):
                X_train = _safe_slice(X, train_idx)
                y_train = _safe_slice(y, train_idx)
                X_val = _safe_slice(X, val_idx)
                
                for est_idx, est in enumerate(self.base_estimators):
                    cloned_est = clone(est)
                    cloned_est.fit(X_train, y_train)
                    val_probs = cloned_est.predict_proba(X_val)[:, 1]
                    oof_predictions[val_idx, est_idx] = val_probs
            
            self.meta_estimator_ = clone(self.meta_estimator).fit(oof_predictions, y)
            # Fit final base estimators on all training data
            self.base_estimators_ = [clone(est).fit(X, y) for est in self.base_estimators]
            
        return self
    
    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        
        base_predictions = np.column_stack([
            est.predict_proba(X)[:, 1] for est in self.base_estimators_
        ])
        return self.meta_estimator_.predict_proba(base_predictions)
    
    def predict(self, X):
        X = validate_data(self, X, reset=False)
        proba = self.predict_proba(X)
        indices = (proba[:, 1] >= 0.5).astype(int)
        return self.classes_[indices]


class StackedEnsembleRegressor(BaseEstimator, RegressorMixin):
    """Stacked ensemble with meta-learner."""
    
    def __init__(self, base_estimators: list, meta_estimator: Any, n_splits: int = 5, random_state: int = RANDOM_STATE):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.n_splits = n_splits
        self.random_state = random_state
    
    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        actual_n_splits = min(self.n_splits, len(X))
        
        if actual_n_splits < 2:
            raise ValueError(
                f"Stacking ensemble requires at least 2-fold cross-validation. "
                f"Got {len(X)} samples. "
                "Increase dataset size or use simple ensemble instead."
            )
        else:
            cv = KFold(n_splits=actual_n_splits, shuffle=True, random_state=self.random_state)
            oof_predictions = np.zeros((len(X), len(self.base_estimators)))
            
            for train_idx, val_idx in cv.split(X, y):
                X_train = _safe_slice(X, train_idx)
                y_train = _safe_slice(y, train_idx)
                X_val = _safe_slice(X, val_idx)
                
                for est_idx, est in enumerate(self.base_estimators):
                    cloned_est = clone(est)
                    cloned_est.fit(X_train, y_train)
                    val_preds = cloned_est.predict(X_val)
                    oof_predictions[val_idx, est_idx] = val_preds
            
            self.meta_estimator_ = clone(self.meta_estimator).fit(oof_predictions, y)
            # Fit final base estimators on all training data
            self.base_estimators_ = [clone(est).fit(X, y) for est in self.base_estimators]
            
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        
        base_predictions = np.column_stack([
            est.predict(X) for est in self.base_estimators_
        ])
        return self.meta_estimator_.predict(base_predictions)


def create_weighted_ensemble_classifier(estimators_dict: dict, weights: dict = None):
    """Create weighted ensemble from model dict."""
    estimators = list(estimators_dict.values())
    weights_list = [weights.get(name, 1.0 / len(estimators)) for name in estimators_dict.keys()] if weights else None
    return WeightedEnsembleClassifier(estimators, weights_list)


def create_weighted_ensemble_regressor(estimators_dict: dict, weights: dict = None):
    """Create weighted ensemble from model dict."""
    estimators = list(estimators_dict.values())
    weights_list = [weights.get(name, 1.0 / len(estimators)) for name in estimators_dict.keys()] if weights else None
    return WeightedEnsembleRegressor(estimators, weights_list)
