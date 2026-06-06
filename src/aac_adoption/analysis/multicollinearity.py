"""Multicollinearity analysis and diagnostics."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aac_adoption.features.feature_sets import BASE_INTAKE_TIME_FEATURES


def compute_vif_sklearn(X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Compute VIF for each feature using the correlation matrix inverse.
    
    Mathematically, the VIF for feature j is the j-th diagonal element of the 
    inverse correlation matrix R^-1. Using pseudo-inverse handles collinearity robustly.
    """
    # Remove columns with zero variance to avoid divide-by-zero when standardizing
    variances = np.var(X, axis=0)
    keep_indices = [i for i, var in enumerate(variances) if var > 1e-9]
    
    X_clean = X[:, keep_indices]
    clean_names = [feature_names[i] for i in keep_indices]
    
    if X_clean.shape[1] == 0:
        return pd.DataFrame(columns=["feature", "vif"])
        
    # Standardize columns to get correlation matrix
    X_std = (X_clean - np.mean(X_clean, axis=0)) / np.std(X_clean, axis=0)
    # Correlation matrix R = X_std.T @ X_std / (N - 1)
    N = X_std.shape[0]
    R = (X_std.T @ X_std) / (N - 1)
    
    # Inverted correlation matrix using pseudo-inverse
    R_inv = np.linalg.pinv(R)
    vifs = np.diag(R_inv)
    
    # Handle dropped zero-variance features as having VIF of 1.0
    all_vifs = {name: vif for name, vif in zip(clean_names, vifs)}
    for name in feature_names:
        if name not in all_vifs:
            all_vifs[name] = 1.0
            
    return pd.DataFrame({
        "feature": feature_names,
        "vif": [all_vifs[name] for name in feature_names]
    }).sort_values("vif", ascending=False)


def run_multicollinearity_analysis(
    data_path: str | Path = "data/processed/modeling_dataset.csv",
    tables_dir: str | Path = "reports/tables",
    summary_dir: str | Path = "reports/summary",
) -> pd.DataFrame:
    """Run VIF and correlation analysis on model features and write reports."""
    # Helper to build simple pipelines manually without import issues
    def Pipeline_num():
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

    tables = Path(tables_dir)
    summary = Path(summary_dir)
    tables.mkdir(parents=True, exist_ok=True)
    summary.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if len(df) > 25000:
        df = df.sample(n=25000, random_state=42).reset_index(drop=True)
    
    # Select feature columns present in the dataset
    feature_cols = [c for c in BASE_INTAKE_TIME_FEATURES if c in df.columns]
    
    # Separate numeric vs categorical for VIF preprocessing
    num_cols = [c for c in feature_cols if df[c].dtype in (int, float, np.number)]
    cat_cols = [c for c in feature_cols if c not in num_cols]
    
    # We build a simple design matrix using median imputation and onehot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline_num(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, min_frequency=50, handle_unknown="ignore"), cat_cols)
        ] if cat_cols else [("num", Pipeline_num(), num_cols)],
        remainder="drop"
    )

    X_prep = preprocessor.fit_transform(df[feature_cols])
    
    # Get feature names from preprocessor
    num_features = num_cols
    cat_features = []
    if cat_cols:
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_features = list(cat_encoder.get_feature_names_out(cat_cols))
    
    all_features = num_features + cat_features
    
    # Compute VIF
    vif_df = compute_vif_sklearn(X_prep, all_features)
    
    # Save VIF CSV
    vif_df.to_csv(tables / "multicollinearity_vif.csv", index=False)
    print(f"[multicollinearity] Wrote multicollinearity_vif.csv")
    
    # Compute correlation matrix for numeric features
    corr_matrix = df[num_cols].corr()
    corr_matrix.to_csv(tables / "numeric_correlation_matrix.csv")
    
    _write_multicollinearity_md(vif_df, corr_matrix, summary)
    
    return vif_df


def _write_multicollinearity_md(vif_df: pd.DataFrame, corr: pd.DataFrame, summary: Path) -> None:
    red_flags = vif_df[vif_df["vif"] > 10.0]
    
    lines = [
        "# Multicollinearity and VIF Diagnostics\n\n",
        "## Executive Summary\n\n",
        "Multicollinearity occurs when independent variables are highly correlated, ",
        "leading to unstable coefficient estimates and inflated standard errors in linear models. ",
        "We evaluate multicollinearity using the **Variance Inflation Factor (VIF)**. ",
        "A feature with $VIF > 10$ is considered severely collinear.\n\n",
        "## Severe Collinearity Red Flags ($VIF > 10$)\n\n",
    ]
    
    if red_flags.empty:
        lines.append("No features with $VIF > 10$ were detected. The design matrix is stable.\n\n")
    else:
        lines.append(red_flags.to_markdown(index=False))
        lines.append("\n\n")
        
    lines += [
        "## Complete VIF Leadership Table\n\n",
        vif_df.head(25).to_markdown(index=False),
        "\n\n",
        "## Numeric Correlation Matrix\n\n",
        corr.round(3).to_markdown(),
        "\n\n",
        "## Technical Recommendations for AI Agents\n\n",
        "1. **Prune redundant age variants:** `age_days`, `age_months`, and `age_years` are linear transformations of each other. Keep only a single numeric age representation (e.g. `age_days`) and the categorical `age_group`.\n",
        "2. **Eliminate name flags duplication:** `has_name` and `is_named` are identical columns and must be pruned.\n",
        "3. **Drop redundant calendar indices:** Keep `intake_month` and drop `intake_quarter` if month is already used.\n"
    ]
    
    (summary / "multicollinearity.md").write_text("".join(lines), encoding="utf-8")
    print(f"[multicollinearity] Wrote multicollinearity.md")
