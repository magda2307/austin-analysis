"""Audit module to detect and prevent target leakage."""

import pandas as pd


LEAKAGE_COLUMNS_TO_AUDIT = {
    "intake_condition",
    "sex_upon_intake",
}


def audit_leakage_columns(df: pd.DataFrame) -> list[str]:
    """Audit dataset for columns that could cause target leakage."""
    found_leakage = []
    
    for col in LEAKAGE_COLUMNS_TO_AUDIT:
        if col in df.columns:
            col_data = df[col]
            if col_data.isna().all():
                found_leakage.append(f"{col}: all values are NaN")
            elif col_data.nunique() <= 1:
                found_leakage.append(f"{col}: constant value (no variation)")
            elif col_data.dtype == "datetime64[ns]":
                found_leakage.append(f"{col}: datetime type (should be categorical)")
    
    if "intake_condition" in df.columns:
        condition_values = df["intake_condition"].dropna().astype(str).str.lower().unique()
        for val in condition_values:
            if any(term in val for term in ["adopt", "adoption", "transfer"]):
                found_leakage.append("intake_condition contains outcome-related values")
                break
    
    if "sex_upon_intake" in df.columns:
        sex_values = df["sex_upon_intake"].dropna().astype(str).str.lower().unique()
        for val in sex_values:
            if any(term in val for term in ["spayed", "neutered"]):
                found_leakage.append("sex_upon_intake contains sterilization status (potential outcome correlate)")
                break
    
    return found_leakage


def drop_leakage_columns(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Remove leakage-prone columns from dataframe."""
    cols_to_check = columns or list(LEAKAGE_COLUMNS_TO_AUDIT)
    result = df.copy()
    for col in cols_to_check:
        if col in result.columns:
            result = result.drop(columns=[col])
    return result
