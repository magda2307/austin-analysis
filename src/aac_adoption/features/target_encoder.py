"""Out-of-Fold Bayesian Target Encoder to prevent target leakage."""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from aac_adoption.features.feature_sets import LEAKAGE_COLUMNS


class OOFBayesianTargetEncoder(BaseEstimator, TransformerMixin):
    """Bayesian target encoder with out-of-fold cross-fitting for training data.

    Applies empirical Bayes shrinkage:
        S_j = (n_j * mean_j + m * global_mean) / (n_j + m)
    where m is the prior weight (smoothing parameter).
    """
    
    HIGH_CARDINALITY_THRESHOLD = 5

    def __init__(
        self,
        columns: list[str],
        smoothing: float = 10.0,
        n_splits: int = 5,
        random_state: int | None = 42,
        handle_unknown: str = "return_nan",
    ):
        self.columns = [c for c in columns if c not in LEAKAGE_COLUMNS]
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state
        self.handle_unknown = handle_unknown
        self.global_mean_ = 0.0
        self.encoding_map_ = {}
        self.high_cardinality_cols_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> OOFBayesianTargetEncoder:
        """Fit target encoder and compute global mappings."""
        self.global_mean_ = float(y.mean())
        self.encoding_map_ = {}
        self.high_cardinality_cols_ = []

        for col in self.columns:
            if col not in X.columns:
                continue
            
            cardinality = X[col].nunique()
            if cardinality > self.HIGH_CARDINALITY_THRESHOLD:
                self.high_cardinality_cols_.append(col)
            
            stats = pd.DataFrame({"feat": X[col], "target": y}).groupby("feat")["target"].agg(["count", "mean"])
            counts = stats["count"]
            means = stats["mean"]
            
            encoded_vals = (counts * means + self.smoothing * self.global_mean_) / (counts + self.smoothing)
            self.encoding_map_[col] = encoded_vals.to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply global mappings to out-of-sample data."""
        X_out = X.copy()
        for col in self.columns:
            if col not in X_out.columns:
                continue
            
            mapping = self.encoding_map_.get(col, {})
            if self.handle_unknown == "return_nan":
                X_out[col] = X_out[col].map(mapping)
            else:
                X_out[col] = X_out[col].map(mapping).fillna(self.global_mean_).astype(float)
            
        return X_out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Compute out-of-fold target encodings for the training set to prevent leakage."""
        self.fit(X, y)
        X_out = X.copy()
        
        # Instantiate stratified K-Fold
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Temp series to hold OOF predictions
        oof_encoded = {col: pd.Series(index=X.index, dtype=float) for col in self.columns if col in X.columns}
        
        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            
            for col in self.columns:
                if col not in X.columns:
                    continue
                
                # Fit temporary encoder on train folds
                fold_global_mean = float(y_tr.mean())
                stats = pd.DataFrame({"feat": X_tr[col], "target": y_tr}).groupby("feat")["target"].agg(["count", "mean"])
                counts = stats["count"]
                means = stats["mean"]
                
                encoded_vals = (counts * means + self.smoothing * fold_global_mean) / (counts + self.smoothing)
                mapping = encoded_vals.to_dict()
                
                # Apply mapping to validation fold
                oof_encoded[col].iloc[val_idx] = X_val[col].map(mapping).fillna(fold_global_mean)
                
        # Update columns in X_out with OOF values
        for col, oof_series in oof_encoded.items():
            # If any missing values (due to unmapped items in K-Fold), fill with global mean
            X_out[col] = oof_series.fillna(self.global_mean_)
            
        return X_out
