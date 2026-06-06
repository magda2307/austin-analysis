"""Rolling features cache for decoupled context features."""

from pathlib import Path
from typing import Optional

import pandas as pd


class RollingFeaturesCache:
    """Cached rolling feature calculations to decouple from main computation."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(".rolling_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._cache = {}
    
    def get_intake_volume(self, dataset_date: pd.Timestamp, window_days: int) -> float:
        """Get cached intake volume for a date/window combination."""
        cache_key = f"intake_volume_{window_days}d"
        if cache_key not in self._cache:
            self._load_cache(cache_key)
        
        cache_series = self._cache[cache_key]
        cutoff_date = dataset_date - pd.Timedelta(days=window_days)
        
        mask = (cache_series.index >= cutoff_date) & (cache_series.index < dataset_date)
        if mask.any():
            return float(cache_series[mask].sum())
        return 0.0
    
    def store_intake_volume(self, date: pd.Timestamp, volume: float, window_days: int) -> None:
        """Store calculated intake volume in cache."""
        cache_key = f"intake_volume_{window_days}d"
        if cache_key not in self._cache:
            self._cache[cache_key] = pd.Series(dtype=float)
        
        self._cache[cache_key].at[date] = volume
    
    def _load_cache(self, key: str) -> None:
        """Load cache from disk if available."""
        cache_file = self.cache_dir / f"{key}.parquet"
        if cache_file.exists():
            self._cache[key] = pd.read_parquet(cache_file)
        else:
            self._cache[key] = pd.Series(dtype=float)


def compute_rolling_features_decoupled(
    intake_daily: pd.DataFrame,
    windows: list[int] = [7, 30],
) -> pd.DataFrame:
    """Compute rolling features independently from main dataset."""
    if intake_daily.empty:
        return pd.DataFrame(columns=["context_date", *[f"intake_volume_{w}d" for w in windows]])
    
    intake_daily = intake_daily.copy()
    intake_daily["context_date"] = pd.to_datetime(intake_daily["context_date"], errors="coerce").dt.normalize()
    intake_daily = intake_daily.dropna(subset=["context_date"])
    
    intake_daily = intake_daily.groupby("context_date")["intake_volume"].sum().reset_index()
    intake_daily = intake_daily.sort_values("context_date")
    
    result = pd.DataFrame({"context_date": intake_daily["context_date"]})
    
    for window in windows:
        result[f"intake_volume_{window}d"] = (
            intake_daily["intake_volume"]
            .shift(1)
            .rolling(window, min_periods=1)
            .sum()
            .fillna(0)
            .values
        )
    
    return result
