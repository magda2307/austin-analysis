"""True time-based rolling window computations for intake history."""

import numpy as np
import pandas as pd


def compute_prior_intake_counts(raw_intakes: pd.DataFrame, windows: list[int] = [7, 30]) -> pd.DataFrame:
    """Compute exact time-based rolling intake volume windows [t - window, t)."""
    df = pd.DataFrame({
        "animal_id": raw_intakes["animal_id"],
        "intake_datetime": pd.to_datetime(raw_intakes["intake_datetime"], errors="coerce")
    })
    valid_mask = df["intake_datetime"].notna()
    valid_df = (
        df[valid_mask]
        .drop_duplicates(["animal_id", "intake_datetime"], keep="first")
        .sort_values(["intake_datetime", "animal_id"], kind="stable")
        .copy()
    )
    
    timestamps = valid_df["intake_datetime"].values
    
    for window in windows:
        start_times = valid_df["intake_datetime"] - pd.Timedelta(days=window)
        start_indices = np.searchsorted(timestamps, start_times.values, side="left")
        end_indices = np.searchsorted(
            timestamps,
            valid_df["intake_datetime"].values,
            side="left",
        )
        counts = end_indices - start_indices
        valid_df[f"intake_volume_{window}d"] = np.maximum(counts, 0)

    feature_columns = [f"intake_volume_{window}d" for window in windows]
    result_df = df.merge(
        valid_df[["animal_id", "intake_datetime", *feature_columns]],
        on=["animal_id", "intake_datetime"],
        how="left",
        sort=False,
    )
    
    for window in windows:
        result_df[f"intake_volume_{window}d"] = result_df[f"intake_volume_{window}d"].fillna(0).astype(int)
        
    return result_df
