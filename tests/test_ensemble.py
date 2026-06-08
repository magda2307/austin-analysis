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


def test_stacked_ensemble_oof_classifier():
    # 1. Create a random dataset that is not easily predictable
    np.random.seed(123)
    X = pd.DataFrame({"feat": np.random.randn(100)})
    y = pd.Series(np.random.choice([0, 1], size=100))
    
    # 2. Define a base estimator that will achieve 100% training accuracy (memorization)
    from sklearn.tree import DecisionTreeClassifier
    base_est = DecisionTreeClassifier(random_state=42)
    
    # Verify first that this base estimator indeed gets 100% training accuracy if trained on full data
    base_est.fit(X, y)
    full_preds = base_est.predict(X)
    assert np.mean(full_preds == y) == 1.0
    
    # 3. Define a custom meta-learner to record the X (OOF predictions) passed to fit
    from sklearn.base import BaseEstimator, ClassifierMixin
    
    recorded_X = []
    
    class RecordingMetaClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, recorder=None):
            self.recorder = recorder
            
        def fit(self, X_meta, y_meta):
            if self.recorder is not None:
                self.recorder(X_meta, y_meta)
            self.classes_ = np.unique(y_meta)
            return self
            
        def predict_proba(self, X_meta):
            return np.column_stack([1.0 - X_meta[:, 0], X_meta[:, 0]])
            
        def predict(self, X_meta):
            return (X_meta[:, 0] >= 0.5).astype(int)
            
    def record_fit(X_meta, y_meta):
        recorded_X.append(X_meta.copy())
        
    meta_est = RecordingMetaClassifier(recorder=record_fit)
    
    ensemble = StackedEnsembleClassifier(
        base_estimators=[DecisionTreeClassifier(random_state=42)],
        meta_estimator=meta_est
    )
    
    # 4. Train the ensemble
    ensemble.fit(X, y)
    
    # 5. Retrieve the recorded inputs to the meta-learner
    assert len(recorded_X) == 1
    meta_train_X = recorded_X[0]
    
    # meta_train_X should contain the out-of-fold probability predictions of the base estimator.
    # Convert these probabilities to binary class predictions.
    oof_predictions = (meta_train_X[:, 0] >= 0.5).astype(int)
    
    # Verify that the out-of-fold predictions do NOT achieve 100% accuracy on the target.
    oof_accuracy = np.mean(oof_predictions == y)
    assert oof_accuracy < 1.0


def test_stacked_ensemble_oof_regressor():
    # 1. Create a random dataset that is not easily predictable
    np.random.seed(123)
    X = pd.DataFrame({"feat": np.random.randn(100)})
    y = pd.Series(np.random.randn(100))
    
    # 2. Define a base estimator that will achieve 0.0 MSE (memorization)
    from sklearn.tree import DecisionTreeRegressor
    base_est = DecisionTreeRegressor(random_state=42)
    
    # Verify first that this base estimator gets ~0 MSE if trained on full data
    base_est.fit(X, y)
    full_preds = base_est.predict(X)
    assert np.mean((full_preds - y) ** 2) < 1e-10
    
    # 3. Define a custom meta-learner to record the X (OOF predictions) passed to fit
    from sklearn.base import BaseEstimator, RegressorMixin
    
    recorded_X = []
    
    class RecordingMetaRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, recorder=None):
            self.recorder = recorder
            
        def fit(self, X_meta, y_meta):
            if self.recorder is not None:
                self.recorder(X_meta, y_meta)
            return self
            
        def predict(self, X_meta):
            return X_meta[:, 0]
            
    def record_fit(X_meta, y_meta):
        recorded_X.append(X_meta.copy())
        
    meta_est = RecordingMetaRegressor(recorder=record_fit)
    
    ensemble = StackedEnsembleRegressor(
        base_estimators=[DecisionTreeRegressor(random_state=42)],
        meta_estimator=meta_est
    )
    
    # 4. Train the ensemble
    ensemble.fit(X, y)
    
    # 5. Retrieve the recorded inputs to the meta-learner
    assert len(recorded_X) == 1
    meta_train_X = recorded_X[0]
    
    # meta_train_X should contain the out-of-fold predictions of the base estimator.
    oof_predictions = meta_train_X[:, 0]
    
    # Verify that the out-of-fold predictions do NOT achieve 0.0 MSE on the target.
    oof_mse = np.mean((oof_predictions - y) ** 2)
    assert oof_mse > 0.01


from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class MockMetaClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.fitted_X_ = None
        self.fitted_y_ = None
        
    def fit(self, X, y):
        self.fitted_X_ = np.array(X)
        self.fitted_y_ = np.array(y)
        self.classes_ = np.unique(y)
        return self
        
    def predict_proba(self, X):
        return np.zeros((len(X), 2))


class MockMetaRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.fitted_X_ = None
        self.fitted_y_ = None
        
    def fit(self, X, y):
        self.fitted_X_ = np.array(X)
        self.fitted_y_ = np.array(y)
        return self
        
    def predict(self, X):
        return np.zeros(len(X))


def test_stacked_ensemble_classifier_oof(sample_data):
    X, y = sample_data
    
    # DecisionTree without limits perfectly memorizes training data in-sample
    base_est = DecisionTreeClassifier(random_state=42)
    base_est.fit(X, y)
    assert np.mean(base_est.predict(X) == y) == 1.0
    
    meta = MockMetaClassifier()
    ensemble = StackedEnsembleClassifier(
        base_estimators=[DecisionTreeClassifier(random_state=42)],
        meta_estimator=meta
    )
    
    ensemble.fit(X, y)
    
    # OOF predictions from base estimators passed to meta-learner
    oof_probs = ensemble.meta_estimator_.fitted_X_[:, 0]
    oof_preds = (oof_probs >= 0.5).astype(int)
    
    # OOF predictions should not be 100% accurate
    oof_accuracy = np.mean(oof_preds == y)
    assert oof_accuracy < 1.0


def test_stacked_ensemble_regressor_oof(sample_data_regression):
    X, y = sample_data_regression
    
    # DecisionTreeRegressor without limits perfectly memorizes training data in-sample
    base_est = DecisionTreeRegressor(random_state=42)
    base_est.fit(X, y)
    assert np.allclose(base_est.predict(X), y)
    
    meta = MockMetaRegressor()
    ensemble = StackedEnsembleRegressor(
        base_estimators=[DecisionTreeRegressor(random_state=42)],
        meta_estimator=meta
    )
    
    ensemble.fit(X, y)
    
    # OOF predictions from base estimators passed to meta-learner
    oof_preds = ensemble.meta_estimator_.fitted_X_[:, 0]
    
    # OOF predictions should not perfectly match y
    assert not np.allclose(oof_preds, y)


def test_stacked_ensemble_classifier_fallback():
    # Test case where stratification is impossible (e.g. one class has only 1 sample)
    np.random.seed(42)
    X = pd.DataFrame({"feature1": np.random.randn(10)})
    # Class 1 has only 1 sample, which is less than actual_n_splits=5
    y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    
    from sklearn.dummy import DummyClassifier
    base_est = DummyClassifier(strategy="prior")
    meta = MockMetaClassifier()
    
    ensemble = StackedEnsembleClassifier(
        base_estimators=[base_est],
        meta_estimator=meta,
        n_splits=5,
        random_state=42
    )
    
    with pytest.raises(ValueError, match="Stacking ensemble requires at least"):
        ensemble.fit(X, y)


def test_stacked_ensemble_regressor_fallback():
    # Test case where actual_n_splits < 2 (e.g. only 1 sample)
    np.random.seed(42)
    X = pd.DataFrame({"feature1": [1.0]})
    y = pd.Series([2.0])
    
    from sklearn.dummy import DummyRegressor
    base_est = DummyRegressor(strategy="mean")
    meta = MockMetaRegressor()
    
    ensemble = StackedEnsembleRegressor(
        base_estimators=[base_est],
        meta_estimator=meta,
        n_splits=5,
        random_state=42
    )
    
    with pytest.raises(ValueError, match="Stacking ensemble requires at least"):
        ensemble.fit(X, y)


def test_weighted_ensemble_classifier_string_labels():
    np.random.seed(42)
    X = pd.DataFrame({
        "feature1": np.random.randn(50),
        "feature2": np.random.randn(50),
    })
    # String labels instead of [0, 1] integers
    y = pd.Series(np.random.choice(["adopt", "no_adopt"], 50))
    
    estimators = [
        RandomForestClassifier(n_estimators=5, random_state=42),
        DummyClassifier(strategy="stratified", random_state=42),
    ]
    
    ensemble = WeightedEnsembleClassifier(
        estimators=estimators,
        weights=[0.6, 0.4],
    )
    
    ensemble.fit(X, y)
    predictions = ensemble.predict(X)
    probabilities = ensemble.predict_proba(X)
    
    assert len(predictions) == 50
    assert probabilities.shape == (50, 2)
    assert set(predictions).issubset({"adopt", "no_adopt"})
    # Check classes_ has the right classes
    assert set(ensemble.classes_) == {"adopt", "no_adopt"}


def test_stacked_ensemble_classifier_string_labels():
    np.random.seed(42)
    X = pd.DataFrame({
        "feature1": np.random.randn(50),
        "feature2": np.random.randn(50),
    })
    y = pd.Series(np.random.choice(["adopt", "no_adopt"], 50))
    
    base_estimators = [
        RandomForestClassifier(n_estimators=5, random_state=42),
        GradientBoostingClassifier(n_estimators=5, random_state=42),
    ]
    
    ensemble = StackedEnsembleClassifier(
        base_estimators=base_estimators,
        meta_estimator=LogisticRegression(max_iter=1000, random_state=42),
        n_splits=3,
        random_state=42
    )
    
    ensemble.fit(X, y)
    predictions = ensemble.predict(X)
    probabilities = ensemble.predict_proba(X)
    
    assert len(predictions) == 50
    assert probabilities.shape == (50, 2)
    assert set(predictions).issubset({"adopt", "no_adopt"})
    assert set(ensemble.classes_) == {"adopt", "no_adopt"}


import pytest
pytestmark = pytest.mark.slow
