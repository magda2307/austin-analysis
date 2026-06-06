"""Isotonic and Platt calibration with holdout validation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from aac_adoption.models.artifacts import artifact_path, save_model_artifact
from aac_adoption.models.evaluate import classification_metrics
from aac_adoption.models.split import make_time_split
from aac_adoption.models.train_baseline import ANIMAL_SUBSETS, limit_rows


@dataclass(frozen=True)
class CalibrationOutputs:
    """Metric tables returned by classifier calibration."""

    classification_metrics: pd.DataFrame


def _calibration_method(method: str) -> str:
    if method in {"platt", "sigmoid"}:
        return "sigmoid"
    if method == "isotonic":
        return "isotonic"
    return "isotonic"


class PrefitProbabilityCalibrator:
    """Calibrate probabilities from an already-fitted binary classifier."""

    def __init__(self, base_model: Any, method: str = "isotonic"):
        self.base_model = base_model
        self.method = _calibration_method(method)

    def fit(self, X_calib: pd.DataFrame, y_calib: pd.Series):
        scores = self.base_model.predict_proba(X_calib)[:, 1]
        if self.method == "sigmoid":
            self.calibrator_ = LogisticRegression(solver="lbfgs")
            self.calibrator_.fit(scores.reshape(-1, 1), y_calib)
        else:
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
            self.calibrator_.fit(scores, y_calib)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        scores = self.base_model.predict_proba(X)[:, 1]
        if self.method == "sigmoid":
            calibrated = self.calibrator_.predict_proba(scores.reshape(-1, 1))[:, 1]
        else:
            calibrated = self.calibrator_.predict(scores)
        calibrated = np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _prefit_calibrator(base_model: Any, method: str) -> PrefitProbabilityCalibrator:
    return PrefitProbabilityCalibrator(base_model, method)


def _split_frame_for_calibration(validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split validation chronologically into early-stop and calibration frames."""
    if validation.empty or len(validation) < 4:
        return validation, validation.iloc[0:0].copy()
    sorted_validation = validation.sort_values("intake_datetime") if "intake_datetime" in validation.columns else validation
    midpoint = len(sorted_validation) // 2
    return sorted_validation.iloc[:midpoint].copy(), sorted_validation.iloc[midpoint:].copy()


def _prepare_frame_for_artifact(model: Any, frame: pd.DataFrame, feature_columns: list[str], metadata: dict[str, Any]) -> pd.DataFrame:
    prepared = frame[feature_columns].copy()
    if model.__class__.__name__.startswith("CatBoost"):
        categorical_features = set(metadata.get("categorical_features", []))
        for column in feature_columns:
            if column in categorical_features:
                prepared[column] = prepared[column].astype("string").fillna("unknown").astype(str)
            else:
                prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared


def calibrate_with_isotonic(
    base_model: Any,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    X_test: pd.DataFrame,
    method: str = "isotonic",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply isotonic regression calibration."""
    calibrated = _prefit_calibrator(base_model, _calibration_method(method))
    calibrated.fit(X_calib, y_calib)
    
    predictions = calibrated.predict(X_test).astype(int)
    scores = calibrated.predict_proba(X_test)[:, 1]
    
    return predictions, scores


def calibrate_with_platt(
    base_model: Any,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Platt scaling (sigmoid) calibration."""
    return calibrate_with_isotonic(base_model, X_calib, y_calib, X_test, method="sigmoid")


def post_hoc_calibration_pipeline(
    base_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    calib_method: str = "isotonic",
) -> tuple[np.ndarray, np.ndarray, PrefitProbabilityCalibrator]:
    """Complete post-hoc calibration pipeline."""
    calibrated = _prefit_calibrator(base_model, _calibration_method(calib_method))
    calibrated.fit(X_val, y_val)
    
    predictions = calibrated.predict(X_test).astype(int)
    scores = calibrated.predict_proba(X_test)[:, 1]
    
    return predictions, scores, calibrated


def apply_calibration_to_predictions(
    base_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    calib_method: str = "isotonic",
) -> PrefitProbabilityCalibrator:
    """Train calibration on calibration set using pre-fitted base model."""
    calibrated = _prefit_calibrator(base_model, _calibration_method(calib_method))
    calibrated.fit(X_calib, y_calib)
    
    return calibrated


def save_calibration(calibrated: PrefitProbabilityCalibrator, path: str) -> None:
    """Save calibrated model to disk."""
    joblib.dump(calibrated, path)


def load_calibration(path: str) -> PrefitProbabilityCalibrator:
    """Load calibrated model from disk."""
    return joblib.load(path)


def calibrate_classifiers(
    *,
    data_path: str | Path,
    source_artifacts: list[tuple[str | Path, str]],
    metrics_dir: str | Path,
    models_dir: str | Path,
    max_rows: int | None = None,
    calib_method: str = "isotonic",
) -> CalibrationOutputs:
    """Calibrate saved classifier artifacts on validation and evaluate on test."""
    header = pd.read_csv(data_path, nrows=0)
    parse_dates = [col for col in ["intake_datetime", "outcome_datetime"] if col in header.columns]
    df = pd.read_csv(data_path, parse_dates=parse_dates)
    df = limit_rows(df, max_rows)

    metrics_output_dir = Path(metrics_dir)
    model_output_dir = Path(models_dir)
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for source_dir, model_name in source_artifacts:
        for subset in ANIMAL_SUBSETS:
            source_path = artifact_path(source_dir, "classification", subset, model_name)
            metadata_path = source_path.with_suffix(".json")
            if not source_path.exists():
                continue

            model = joblib.load(source_path)
            metadata: dict[str, Any] = {}
            if metadata_path.exists():
                import json

                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            split = make_time_split(df, "classification_target", animal_subset=subset)
            _, calibration_validation = _split_frame_for_calibration(split.validation)
            if calibration_validation.empty or split.test.empty:
                continue

            feature_columns = metadata.get("feature_columns")
            if not feature_columns:
                feature_columns = [column for column in split.train.columns if column not in {"classification_target"}]
            feature_columns = [column for column in feature_columns if column in calibration_validation.columns]

            val_x = _prepare_frame_for_artifact(model, calibration_validation, feature_columns, metadata)
            test_x = _prepare_frame_for_artifact(model, split.test, feature_columns, metadata)
            calibrated = apply_calibration_to_predictions(
                base_model=model,
                X_train=_prepare_frame_for_artifact(model, split.train, feature_columns, metadata),
                y_train=split.train["classification_target"],
                X_calib=val_x,
                y_calib=calibration_validation["classification_target"],
                calib_method=calib_method,
            )

            calibrated_model_name = f"{model_name}_calibrated"
            calibrated_metadata = {
                **metadata,
                "model_name": calibrated_model_name,
                "task": "classification_calibrated",
                "base_model_name": model_name,
                "base_artifact_path": str(source_path),
                "calibration_method": _calibration_method(calib_method),
                "calibration_rows": len(calibration_validation),
                "feature_columns": feature_columns,
            }
            calibrated_metadata["artifact_path"] = str(
                artifact_path(
                    model_output_dir,
                    "classification_calibrated",
                    subset,
                    calibrated_model_name,
                )
            )
            calibrated_path = save_model_artifact(
                calibrated,
                model_output_dir,
                "classification_calibrated",
                subset,
                calibrated_model_name,
                calibrated_metadata,
            )

            predictions = calibrated.predict(test_x).astype(int)
            scores = calibrated.predict_proba(test_x)[:, 1]
            metrics = classification_metrics(
                split.test["classification_target"],
                predictions,
                scores,
                compute_ci=False,
            )
            rows.append(
                {
                    **calibrated_metadata,
                    "animal_subset": split.animal_subset,
                    "task": "classification_calibrated",
                    "split_strategy": split.strategy,
                    "train_rows": len(split.train),
                    "validation_rows": len(split.validation),
                    "test_rows": len(split.test),
                    **metrics,
                }
            )

    classification = pd.DataFrame(rows)
    classification.to_csv(metrics_output_dir / "calibrated_classification_metrics.csv", index=False)
    return CalibrationOutputs(classification_metrics=classification)
