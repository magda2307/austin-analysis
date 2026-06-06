"""Tests for ensemble methods."""

import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from aac_adoption.models.ensemble import (
    WeightedEnsembleClassifier,
    StackedEnsembleClassifier,
    create_weighted_ensemble_classifier,
    WeightedEnsembleRegressor,
    StackedEnsembleRegressor,
    create_weighted_ensemble_regressor,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "feature3": np.random.choice([0, 1, 2], 100).astype(float),
    })
    y = pd.Series(np.random.choice([0, 1], 100))
    return X, y


@pytest.fixture
def sample_data_regression():
    np.random.seed(42)
    X = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "feature3": np.random.choice([0, 1, 2], 100).astype(float),
    })
    y = pd.Series(np.random.randn(100) * 10 + 20)
    return X, y


def test_weighted_ensemble_classifier(sample_data):
    X, y = sample_data
    
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ("dummy", DummyClassifier(strategy="stratified", random_state=42)),
    ]
    
    ensemble = WeightedEnsembleClassifier(
        estimators=[e[1] for e in estimators],
        weights=[0.7, 0.3],
    )
    
    ensemble.fit(X, y)
    predictions = ensemble.predict(X)
    probabilities = ensemble.predict_proba(X)
    
    assert len(predictions) == 100
    assert probabilities.shape == (100, 2)
    assert set(predictions).issubset({0, 1})


def test_weighted_ensemble_from_dict(sample_data):
    X, y = sample_data
    
    estimators_dict = {
        "rf": RandomForestClassifier(n_estimators=5, random_state=42),
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
    }
    
    ensemble = create_weighted_ensemble_classifier(estimators_dict, weights={"rf": 0.6, "logreg": 0.4})
    
    ensemble.fit(X, y)
    predictions = ensemble.predict(X)
    
    assert len(predictions) == 100


def test_stacked_ensemble_classifier(sample_data):
    X, y = sample_data
    
    base_estimators = [
        RandomForestClassifier(n_estimators=5, random_state=42),
        GradientBoostingClassifier(n_estimators=5, random_state=42),
    ]
    
    ensemble = StackedEnsembleClassifier(
        base_estimators=base_estimators,
        meta_estimator=LogisticRegression(max_iter=1000, random_state=42),
    )
    
    ensemble.fit(X, y)
    predictions = ensemble.predict(X)
    
    assert len(predictions) == 100


def test_weighted_ensemble_equal_weights(sample_data):
    X, y = sample_data
    
    estimators = [
        DummyClassifier(strategy="most_frequent", random_state=42),
        DummyClassifier(strategy="most_frequent", random_state=43),
    ]
    
    ensemble = WeightedEnsembleClassifier(estimators, weights=[0.5, 0.5])
    ensemble.fit(X, y)
    
    assert len(ensemble.estimators_) == 2


def test_weighted_ensemble_regressor(sample_data_regression):
    X, y = sample_data_regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.dummy import DummyRegressor
    
    estimators = [
        ("rf", RandomForestRegressor(n_estimators=5, random_state=42)),
        ("dummy", DummyRegressor(strategy="mean")),
    ]
    
    ensemble = WeightedEnsembleRegressor(
        estimators=[e[1] for e in estimators],
        weights=[0.7, 0.3],
    )
    
    ensemble.fit(X, y)
    predictions = ensemble.predict(X)
    
    assert len(predictions) == 100
    assert not np.isnan(predictions).any()


def test_stacked_ensemble_regressor(sample_data_regression):
    X, y = sample_data_regression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    
    base_estimators = [
        RandomForestRegressor(n_estimators=5, random_state=42),
        GradientBoostingRegressor(n_estimators=5, random_state=42),
    ]
    
    ensemble = StackedEnsembleRegressor(
        base_estimators=base_estimators,
        meta_estimator=LinearRegression(),
    )
    
    ensemble.fit(X, y)
    predictions = ensemble.predict(X)
    
    assert len(predictions) == 100
    assert not np.isnan(predictions).any()
