"""Ensemble methods for combining multiple models."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted


class WeightedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Weighted ensemble of classifiers with calibrated probabilities."""
    
    def __init__(self, estimators: list, weights: list[float] = None):
        self.estimators = estimators
        self.weights = weights or [1.0 / len(estimators)] * len(estimators)
    
    def fit(self, X, y):
        self.estimators_ = []
        for est in self.estimators:
            self.estimators_.append(clone(est).fit(X, y))
        return self
    
    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        all_probs = np.array([est.predict_proba(X) for est in self.estimators_])
        weighted_probs = np.average(all_probs, axis=0, weights=self.weights)
        return weighted_probs
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class WeightedEnsembleRegressor(BaseEstimator, RegressorMixin):
    """Weighted ensemble of regressors."""
    
    def __init__(self, estimators: list, weights: list[float] = None):
        self.estimators = estimators
        self.weights = weights or [1.0 / len(estimators)] * len(estimators)
    
    def fit(self, X, y):
        self.estimators_ = []
        for est in self.estimators:
            self.estimators_.append(clone(est).fit(X, y))
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        all_preds = np.array([est.predict(X) for est in self.estimators_])
        weighted_preds = np.average(all_preds, axis=0, weights=self.weights)
        return weighted_preds


class StackedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Stacked ensemble with meta-learner."""
    
    def __init__(self, base_estimators: list, meta_estimator: Any):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
    
    def fit(self, X, y):
        self.base_estimators_ = [clone(est).fit(X, y) for est in self.base_estimators]
        
        base_predictions = np.column_stack([
            est.predict_proba(X)[:, 1] for est in self.base_estimators_
        ])
        self.meta_estimator_ = clone(self.meta_estimator).fit(base_predictions, y)
        return self
    
    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        base_predictions = np.column_stack([
            est.predict_proba(X)[:, 1] for est in self.base_estimators_
        ])
        return self.meta_estimator_.predict_proba(base_predictions)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class StackedEnsembleRegressor(BaseEstimator, RegressorMixin):
    """Stacked ensemble with meta-learner."""
    
    def __init__(self, base_estimators: list, meta_estimator: Any):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
    
    def fit(self, X, y):
        self.base_estimators_ = [clone(est).fit(X, y) for est in self.base_estimators]
        
        base_predictions = np.column_stack([
            est.predict(X) for est in self.base_estimators_
        ])
        self.meta_estimator_ = clone(self.meta_estimator).fit(base_predictions, y)
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
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
