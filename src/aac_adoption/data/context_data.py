"""External context data for AAC intake-time feature enrichment."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from aac_adoption.features.rolling_features_cache import compute_prior_intake_counts


NCEI_DAILY_SUMMARIES_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
AUSTIN_311_DATASET_ID = "xwdj-i9he"
AUSTIN_311_URL = f"https://data.austintexas.gov/resource/{AUSTIN_311_DATASET_ID}.csv"
DEFAULT_AUSTIN_WEATHER_STATION = "USW00013958"

CONTEXT_FEATURES = [
    "daily_temp_max",
    "daily_temp_min",
    "daily_precipitation",
    "is_extreme_heat",
    "is_rainy_day",
    "weather_available",
    "animal_311_requests_7d",
    "animal_311_requests_30d",
    "intake_volume_7d",
    "intake_volume_30d",
]

NUMERIC_CONTEXT_FEATURES = [
    "daily_temp_max",
    "daily_temp_min",
    "daily_precipitation",
    "animal_311_requests_7d",
    "animal_311_requests_30d",
    "intake_volume_7d",
    "intake_volume_30d",
]

CATEGORICAL_CONTEXT_FEATURES = [
    "is_extreme_heat",
    "is_rainy_day",
    "weather_available",
]


def _download_csv(url: str, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "aac-adoption-thesis/0.1"})
    with urlopen(request, timeout=120) as response:
        output.write_bytes(response.read())
    return output


def download_weather_daily(
    output_path: str | Path,
    *,
    start_date: str = "2013-10-01",
    end_date: str = "2025-05-05",
    station: str = DEFAULT_AUSTIN_WEATHER_STATION,
) -> Path:
    """Download Austin daily weather summaries from NOAA/NCEI."""
    params = {
        "dataset": "daily-summaries",
        "stations": station,
        "startDate": start_date,
        "endDate": end_date,
        "dataTypes": "TMAX,TMIN,PRCP",
        "format": "csv",
        "units": "standard",
        "includeAttributes": "false",
    }
    return _download_csv(f"{NCEI_DAILY_SUMMARIES_URL}?{urlencode(params)}", output_path)


def download_austin_311_animal_requests(
    output_path: str | Path,
    *,
    start_date: str = "2013-10-01",
    end_date: str = "2025-05-05",
) -> Path:
    """Download daily Austin 311 animal-service request counts."""
    params = {
        "$select": "date_trunc_ymd(sr_created_date) as request_date, count(*) as animal_311_requests",
        "$where": (
            f"sr_created_date between '{start_date}T00:00:00' and '{end_date}T23:59:59' "
            "and sr_department_desc = 'Animal Services Office'"
        ),
        "$group": "date_trunc_ymd(sr_created_date)",
        "$order": "request_date",
        "$limit": "50000",
    }
    return _download_csv(f"{AUSTIN_311_URL}?{urlencode(params)}", output_path)


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {column.lower().strip().replace(" ", "_"): column for column in df.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def normalize_weather_daily(weather: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw NOAA-style daily weather data to context columns."""
    if weather.empty:
        return pd.DataFrame(columns=["context_date", "daily_temp_max", "daily_temp_min", "daily_precipitation"])

    date_col = _first_existing_column(weather, ["date", "context_date"])
    max_col = _first_existing_column(weather, ["tmax", "daily_temp_max"])
    min_col = _first_existing_column(weather, ["tmin", "daily_temp_min"])
    precip_col = _first_existing_column(weather, ["prcp", "daily_precipitation"])
    if date_col is None:
        raise ValueError("weather context missing date column")

    result = pd.DataFrame({"context_date": pd.to_datetime(weather[date_col], errors="coerce", format="mixed").dt.normalize()})
    result["daily_temp_max"] = pd.to_numeric(weather[max_col], errors="coerce") if max_col else pd.NA
    result["daily_temp_min"] = pd.to_numeric(weather[min_col], errors="coerce") if min_col else pd.NA
    result["daily_precipitation"] = pd.to_numeric(weather[precip_col], errors="coerce") if precip_col else pd.NA
    return result.dropna(subset=["context_date"]).drop_duplicates("context_date", keep="last")


def normalize_311_animal_requests(requests: pd.DataFrame) -> pd.DataFrame:
    """Normalize 311 animal-service request data to daily counts."""
    if requests.empty:
        return pd.DataFrame(columns=["context_date", "animal_311_requests"])

    date_col = _first_existing_column(requests, ["request_date", "created_date", "context_date"])
    count_col = _first_existing_column(requests, ["animal_311_requests", "count"])
    if date_col is None:
        raise ValueError("311 context missing request date column")

    result = requests.copy()
    if count_col is None:
        description_col = _first_existing_column(result, ["sr_description", "service_request_type", "issue_description"])
        if description_col:
            result = result[result[description_col].fillna("").astype(str).str.contains("animal", case=False)]
        daily = (
            pd.DataFrame({"context_date": pd.to_datetime(result[date_col], errors="coerce", format="mixed").dt.normalize()})
            .dropna(subset=["context_date"])
            .groupby("context_date")
            .size()
            .reset_index(name="animal_311_requests")
        )
        return daily

    daily = pd.DataFrame(
        {
            "context_date": pd.to_datetime(result[date_col], errors="coerce", format="mixed").dt.normalize(),
            "animal_311_requests": pd.to_numeric(result[count_col], errors="coerce").fillna(0),
        }
    )
    return (
        daily.dropna(subset=["context_date"])
        .groupby("context_date", as_index=False)["animal_311_requests"]
        .sum()
    )


def _rolling_prior_counts(
    daily: pd.DataFrame,
    value_column: str,
    windows: list[int],
    target_dates: pd.Series | None = None,
) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["context_date", *[f"{value_column}_{window}d" for window in windows]])
    series = (
        daily[["context_date", value_column]]
        .copy()
        .assign(context_date=lambda df: pd.to_datetime(df["context_date"], format="mixed").dt.normalize())
        .sort_values("context_date")
        .set_index("context_date")[value_column]
        .astype(float)
    )
    min_date = series.index.min()
    max_date = series.index.max()
    if target_dates is not None and not target_dates.dropna().empty:
        target = pd.to_datetime(target_dates.dropna(), format="mixed").dt.normalize()
        min_date = min(min_date, target.min())
        max_date = max(max_date, target.max())
    full_index = pd.date_range(min_date, max_date, freq="D")
    series = series.groupby(level=0).sum().reindex(full_index, fill_value=0.0)
    out = pd.DataFrame({"context_date": series.index})
    for window in windows:
        out[f"{value_column}_{window}d"] = series.shift(1).rolling(window, min_periods=1).sum().fillna(0).values
    return out


def add_context_features(
    modeling_df: pd.DataFrame,
    *,
    raw_intakes: pd.DataFrame,
    weather_daily: pd.DataFrame | None,
    requests_311: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add external and internal prior-window context features."""
    result = modeling_df.copy()
    intake_dates = pd.to_datetime(result["intake_datetime"], errors="coerce").dt.normalize()
    result["context_date"] = intake_dates

    weather = normalize_weather_daily(weather_daily if weather_daily is not None else pd.DataFrame())
    weather = weather.rename(columns={"context_date": "weather_context_date"})
    result["weather_context_date"] = intake_dates - pd.Timedelta(days=1)
    result = result.merge(weather, on="weather_context_date", how="left")

    requests = normalize_311_animal_requests(requests_311 if requests_311 is not None else pd.DataFrame())
    request_rollups = _rolling_prior_counts(requests, "animal_311_requests", [7, 30], target_dates=intake_dates)
    result = result.merge(request_rollups, on="context_date", how="left")

    # Time-based intake volume
    intake_volumes = compute_prior_intake_counts(raw_intakes, [7, 30])
    
    # We join by 'animal_id' and 'intake_datetime'
    # To avoid duplicates if there are identical rows, we drop exact duplicates of keys + features
    join_cols = ["animal_id", "intake_datetime"]
    feat_cols = ["intake_volume_7d", "intake_volume_30d"]
    intake_volumes_clean = intake_volumes[join_cols + feat_cols].dropna(subset=["intake_datetime"]).drop_duplicates(subset=join_cols)
    
    # ensure modeling_df types match for join
    result["_join_datetime"] = pd.to_datetime(result["intake_datetime"], errors="coerce")
    intake_volumes_clean["_join_datetime"] = pd.to_datetime(intake_volumes_clean["intake_datetime"], errors="coerce")
    
    result = result.merge(
        intake_volumes_clean[["animal_id", "_join_datetime", "intake_volume_7d", "intake_volume_30d"]],
        on=["animal_id", "_join_datetime"],
        how="left"
    )
    missing_history = result[["intake_volume_7d", "intake_volume_30d"]].isna().any(axis=1)
    if missing_history.any():
        raise ValueError(
            f"{int(missing_history.sum())} modeling rows are missing from raw intake history"
        )

    weather_columns = ["daily_temp_max", "daily_temp_min", "daily_precipitation"]
    result["weather_available"] = result[weather_columns].notna().any(axis=1)
    
    result["is_extreme_heat"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    max_temp_available = result["daily_temp_max"].notna()
    result.loc[max_temp_available, "is_extreme_heat"] = (
        result.loc[max_temp_available, "daily_temp_max"] >= 95
    )

    result["is_rainy_day"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    precip_avail = result["daily_precipitation"].notna()
    result.loc[precip_avail, "is_rainy_day"] = result.loc[precip_avail, "daily_precipitation"] > 0

    for column in ["animal_311_requests_7d", "animal_311_requests_30d"]:
        result[column] = result[column].fillna(0).astype(float)
    for column in ["intake_volume_7d", "intake_volume_30d"]:
        result[column] = result[column].astype(float)

    result = result.drop(columns=["context_date", "_join_datetime", "weather_context_date"], errors="ignore")
    return result


def add_context_features_from_dir(
    modeling_df: pd.DataFrame, 
    raw_intakes: pd.DataFrame,
    context_data_dir: str | Path,
) -> pd.DataFrame:
    """Load cached context CSVs from a directory and enrich dataset."""
    context_dir = Path(context_data_dir)
    weather_path = context_dir / "austin_weather_daily.csv"
    requests_path = context_dir / "austin_311_animal_requests.csv"
    weather = pd.read_csv(weather_path) if weather_path.exists() else pd.DataFrame()
    requests = pd.read_csv(requests_path) if requests_path.exists() else pd.DataFrame()
    return add_context_features(
        modeling_df, 
        raw_intakes=raw_intakes, 
        weather_daily=weather, 
        requests_311=requests
    )
