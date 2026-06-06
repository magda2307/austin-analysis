"""Hyperparameter tuning with proper cross-validation."""

from typing import Any

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from aac_adoption.config import RANDOM_STATE
from aac_adoption.features.feature_sets import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    available_features_for_df,
)
from aac_adoption.models.split import DatasetSplit


def make_boosting_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing for HistGradientBoosting."""
    numeric_features = available_features_for_df(df, NUMERIC_FEATURES)
    categorical_features = available_features_for_df(df, CATEGORICAL_FEATURES)
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("as_object", FunctionTransformer(lambda x: x.astype(object), feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        sparse_threshold=0.0,
    )


def tune_histgradient_boosting_classification(
    df: pd.DataFrame,
    target_column: str = "classification_target",
    n_splits: int = 3,
    max_iter_options: list[int] = [50, 100, 200],
    max_leaf_nodes_options: list[int] = [15, 31, 63],
    learning_rate_options: list[float] = [0.05, 0.08, 0.1],
) -> dict[str, Any]:
    """Tune HistGradientBoostingClassifier using time-series cross-validation."""
    split = DatasetSplit(
        full_data=df,
        train=df,
        validation=pd.DataFrame(),
        test=pd.DataFrame(),
        strategy="tune",
        train_period="all_train",
        validation_period="",
        test_period="",
        animal_subset="combined",
    )
    
    feature_columns = list(split.train.columns)
    if target_column in feature_columns:
        feature_columns.remove(target_column)
    
    preprocessor = make_boosting_preprocessor(split.train[feature_columns])
    
    best_score = -float("inf")
    best_params = None
    best_depth = None
    
    for max_iter in max_iter_options:
        for max_leaf_nodes in max_leaf_nodes_options:
            for learning_rate in learning_rate_options:
                model = Pipeline(
                    steps=[
                        ("preprocess", preprocessor),
                        (
                            "model",
                            HistGradientBoostingClassifier(
                                max_iter=max_iter,
                                max_leaf_nodes=max_leaf_nodes,
                                learning_rate=learning_rate,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                )
                
                tscv = TimeSeriesSplit(n_splits=n_splits)
                scores = []
                
                try:
                    for train_idx, val_idx in tscv.split(split.train):
                        train_fold = split.train.iloc[train_idx]
                        val_fold = split.train.iloc[val_idx]
                        
                        model.fit(train_fold[feature_columns], train_fold[target_column])
                        score = model.score(val_fold[feature_columns], val_fold[target_column])
                        scores.append(score)
                    
                    mean_score = sum(scores) / len(scores)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {
                            "max_iter": max_iter,
                            "max_leaf_nodes": max_leaf_nodes,
                            "learning_rate": learning_rate,
                        }
                except Exception:
                    continue
    
    return {
        "best_score": best_score,
        "best_params": best_params,
    }


def tune_histgradient_boosting_regression(
    df: pd.DataFrame,
    target_column: str = "regression_target_days",
    n_splits: int = 3,
    max_iter_options: list[int] = [50, 100, 200],
    max_leaf_nodes_options: list[int] = [15, 31, 63],
    learning_rate_options: list[float] = [0.05, 0.08, 0.1],
) -> dict[str, Any]:
    """Tune HistGradientBoostingRegressor using time-series cross-validation."""
    split = DatasetSplit(
        full_data=df,
        train=df,
        validation=pd.DataFrame(),
        test=pd.DataFrame(),
        strategy="tune",
        train_period="all_train",
        validation_period="",
        test_period="",
        animal_subset="combined",
    )
    
    feature_columns = list(split.train.columns)
    if target_column in feature_columns:
        feature_columns.remove(target_column)
    
    preprocessor = make_boosting_preprocessor(split.train[feature_columns])
    
    best_score = -float("inf")
    best_params = None
    best_depth = None
    
    for max_iter in max_iter_options:
        for max_leaf_nodes in max_leaf_nodes_options:
            for learning_rate in learning_rate_options:
                model = Pipeline(
                    steps=[
                        ("preprocess", preprocessor),
                        (
                            "model",
                            HistGradientBoostingRegressor(
                                max_iter=max_iter,
                                max_leaf_nodes=max_leaf_nodes,
                                learning_rate=learning_rate,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                )
                
                tscv = TimeSeriesSplit(n_splits=n_splits)
                scores = []
                
                try:
                    for train_idx, val_idx in tscv.split(split.train):
                        train_fold = split.train.iloc[train_idx]
                        val_fold = split.train.iloc[val_idx]
                        
                        model.fit(train_fold[feature_columns], train_fold[target_column])
                        pred = model.predict(val_fold[feature_columns])
                        mae = abs(pred - val_fold[target_column]).mean()
                        scores.append(-mae)
                    
                    mean_score = sum(scores) / len(scores)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {
                            "max_iter": max_iter,
                            "max_leaf_nodes": max_leaf_nodes,
                            "learning_rate": learning_rate,
                        }
                except Exception:
                    continue
    
    return {
        "best_score": best_score,
        "best_params": best_params,
    }
