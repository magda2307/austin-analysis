"""True time-based rolling window computations for intake history."""

import numpy as np
import pandas as pd


def compute_prior_intake_counts(raw_intakes: pd.DataFrame, windows: list[int] = [7, 30]) -> pd.DataFrame:
    """Compute exact time-based rolling intake volume windows [t - window, t)."""
    df = pd.DataFrame({
        "animal_id": raw_intakes["animal_id"],
        "intake_datetime": pd.to_datetime(raw_intakes["intake_datetime"], errors="coerce")
    })
    df["_original_idx"] = np.arange(len(df))
    valid_mask = df["intake_datetime"].notna()
    valid_df = df[valid_mask].sort_values(["intake_datetime", "_original_idx"]).copy()
    
    timestamps = valid_df["intake_datetime"].values
    
    result_df = df.copy()
    for window in windows:
        start_times = valid_df["intake_datetime"] - pd.Timedelta(days=window)
        start_indices = np.searchsorted(timestamps, start_times.values, side="left")
        end_indices = np.arange(len(valid_df))
        counts = end_indices - start_indices
        valid_df[f"intake_volume_{window}d"] = np.maximum(counts, 0)
        
    result_df = result_df.join(valid_df[[f"intake_volume_{w}d" for w in windows]])
    
    for window in windows:
        result_df[f"intake_volume_{window}d"] = result_df[f"intake_volume_{window}d"].fillna(0).astype(int)
        
    result_df = result_df.drop(columns=["_original_idx"], errors="ignore")
    return result_df
