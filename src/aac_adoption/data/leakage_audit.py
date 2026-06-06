"""Audit module to detect and prevent target leakage."""

import pandas as pd


LEAKAGE_COLUMNS_TO_AUDIT = {
    "intake_condition",
    "sex_upon_intake",
}


class DataLeakageError(ValueError):
    """Exception raised when an unsafe column is found in the audit."""
    pass

def audit_leakage_columns(df: pd.DataFrame) -> dict[str, str]:
    """Audit dataset for columns that could cause target leakage.
    Returns mapping of column names to risk levels: safe, probably_safe, needs_audit, unsafe.
    """
    risk_levels = {col: "safe" for col in df.columns}
    
    for col in LEAKAGE_COLUMNS_TO_AUDIT:
        if col in df.columns:
            col_data = df[col]
            if col_data.isna().all():
                risk_levels[col] = "unsafe"
            elif col_data.nunique() <= 1:
                risk_levels[col] = "unsafe"
            elif col_data.dtype == "datetime64[ns]":
                risk_levels[col] = "unsafe"
            else:
                risk_levels[col] = "probably_safe"
    
    if "intake_condition" in df.columns:
        condition_values = df["intake_condition"].dropna().astype(str).str.lower().unique()
        for val in condition_values:
            if any(term in val for term in ["adopt", "adoption", "transfer"]):
                risk_levels["intake_condition"] = "needs_audit"
                break
    
    if "sex_upon_intake" in df.columns:
        sex_values = df["sex_upon_intake"].dropna().astype(str).str.lower().unique()
        for val in sex_values:
            if any(term in val for term in ["spayed", "neutered"]):
                risk_levels["sex_upon_intake"] = "needs_audit"
                break
    
    unsafe_cols = [c for c, r in risk_levels.items() if r == "unsafe"]
    if unsafe_cols:
        raise DataLeakageError(f"Unsafe leakage columns detected: {unsafe_cols}")
        
    return risk_levels


def drop_leakage_columns(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Remove leakage-prone columns from dataframe."""
    cols_to_check = columns or list(LEAKAGE_COLUMNS_TO_AUDIT)
    result = df.copy()
    for col in cols_to_check:
        if col in result.columns:
            result = result.drop(columns=[col])
    return result
