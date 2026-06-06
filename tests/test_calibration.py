"""Tests for calibration module."""

import pandas as pd
import numpy as np
import pytest
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from aac_adoption.models.calibrate import (
    calibrate_with_isotonic,
    calibrate_with_platt,
    post_hoc_calibration_pipeline,
    apply_calibration_to_predictions,
    calibrate_classifiers,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_train = 50
    n_calib = 30
    n_test = 20
    
    X_train = pd.DataFrame({
        "feature1": np.random.randn(n_train),
        "feature2": np.random.randn(n_train),
    })
    y_train = pd.Series(np.random.choice([0, 1], n_train))
    
    X_calib = pd.DataFrame({
        "feature1": np.random.randn(n_calib),
        "feature2": np.random.randn(n_calib),
    })
    y_calib = pd.Series(np.random.choice([0, 1], n_calib))
    
    X_test = pd.DataFrame({
        "feature1": np.random.randn(n_test),
        "feature2": np.random.randn(n_test),
    })
    
    return X_train, y_train, X_calib, y_calib, X_test


def test_calibrate_with_isotonic(sample_data):
    X_train, y_train, X_calib, y_calib, X_test = sample_data
    
    base_model = RandomForestClassifier(n_estimators=5, random_state=42)
    base_model.fit(X_train, y_train)
    
    predictions, scores = calibrate_with_isotonic(
        base_model,
        X_calib,
        y_calib,
        X_test,
        method="isotonic",
    )
    
    assert len(predictions) == 20
    assert len(scores) == 20
    assert set(predictions).issubset({0, 1})
    assert all(0 <= s <= 1 for s in scores)


def test_calibrate_with_platt(sample_data):
    X_train, y_train, X_calib, y_calib, X_test = sample_data
    
    base_model = RandomForestClassifier(n_estimators=5, random_state=42)
    base_model.fit(X_train, y_train)
    
    predictions, scores = calibrate_with_platt(
        base_model,
        X_calib,
        y_calib,
        X_test,
    )
    
    assert len(predictions) == 20
    assert len(scores) == 20


def test_post_hoc_calibration_pipeline(sample_data):
    X_train, y_train, X_calib, y_calib, X_test = sample_data
    
    base_model = RandomForestClassifier(n_estimators=5, random_state=42)
    base_model.fit(X_train, y_train)
    
    predictions, scores, calibrated = post_hoc_calibration_pipeline(
        base_model,
        X_train,
        y_train,
        X_calib,
        y_calib,
        X_test,
        calib_method="isotonic",
    )
    
    assert len(predictions) == 20
    assert calibrated is not None


def test_apply_calibration_to_predictions(sample_data):
    X_train, y_train, X_calib, y_calib, X_test = sample_data
    
    base_model = RandomForestClassifier(n_estimators=5, random_state=42)
    base_model.fit(X_train, y_train)
    
    calibrated = apply_calibration_to_predictions(
        base_model,
        X_train,
        y_train,
        X_calib,
        y_calib,
        calib_method="isotonic",
    )
    
    predictions = calibrated.predict(X_calib)
    assert len(predictions) == 30


def test_apply_calibration_preserves_platt_method(sample_data):
    X_train, y_train, X_calib, y_calib, X_test = sample_data

    base_model = RandomForestClassifier(n_estimators=5, random_state=42)
    base_model.fit(X_train, y_train)

    calibrated = apply_calibration_to_predictions(
        base_model,
        X_train,
        y_train,
        X_calib,
        y_calib,
        calib_method="platt",
    )

    assert calibrated.method == "sigmoid"


def test_calibrate_classifiers_handles_missing_artifacts(tmp_path, sample_data):
    X_train, y_train, X_calib, y_calib, X_test = sample_data
    data = pd.concat([X_train, X_calib, X_test], ignore_index=True)
    data["animal_type"] = ["Dog"] * len(data)
    data["intake_datetime"] = pd.date_range("2019-01-01", periods=len(data), freq="30D")
    data["outcome_datetime"] = data["intake_datetime"] + pd.Timedelta(days=7)
    data["intake_year"] = data["intake_datetime"].dt.year
    data["classification_target"] = pd.concat([y_train, y_calib, pd.Series(np.random.choice([0, 1], len(X_test)))], ignore_index=True)
    data_path = tmp_path / "modeling.csv"
    data.to_csv(data_path, index=False)

    outputs = calibrate_classifiers(
        data_path=data_path,
        source_artifacts=[(tmp_path / "missing", "catboost")],
        metrics_dir=tmp_path / "metrics",
        models_dir=tmp_path / "models",
    )

    assert outputs.classification_metrics.empty
    assert (tmp_path / "metrics" / "calibrated_classification_metrics.csv").exists()
