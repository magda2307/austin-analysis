"""Initial AAC data cleaning."""

import pandas as pd

from aac_adoption.config import SUPPORTED_ANIMAL_TYPES


def require_columns(df: pd.DataFrame, required_columns: set[str], frame_name: str) -> None:
    """Raise clear error when expected raw columns are missing."""
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{frame_name} missing required columns: {missing}")


def parse_datetime_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Parse datetime columns that exist in the frame.

    Austin historical outcomes include timezone offsets while intakes do not.
    For LOS matching we compare local shelter timestamps, so timezone-aware
    values are made timezone-naive without shifting the clock time.
    """
    result = df.copy()
    for column in columns:
        if column in result.columns:
            normalized = (
                result[column]
                .astype("string")
                .str.strip()
                .str.replace("T", " ", regex=False)
                .str.replace(r"(Z|[+-]\d{2}:?\d{2})$", "", regex=True)
            )
            result[column] = pd.to_datetime(normalized, errors="coerce", format="mixed")
    return result


def filter_cats_and_dogs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep dog and cat records only."""
    if "animal_type" not in df.columns:
        raise ValueError("animal_type column is required")
    mask = df["animal_type"].astype(str).str.strip().str.lower().isin(SUPPORTED_ANIMAL_TYPES)
    return df.loc[mask].copy()


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove only exact duplicate rows."""
    return df.drop_duplicates().copy()


def clean_intakes(df: pd.DataFrame) -> pd.DataFrame:
    """Clean intake data using only safe transformations."""
    require_columns(
        df,
        {"animal_id", "animal_type", "intake_datetime"},
        "intakes",
    )
    result = remove_exact_duplicates(df)
    result = parse_datetime_columns(result, ["intake_datetime"])
    result = result.dropna(subset=["animal_id", "animal_type", "intake_datetime"])
    result = filter_cats_and_dogs(result)
    return result


def clean_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Clean outcome data using only safe transformations."""
    require_columns(
        df,
        {"animal_id", "animal_type", "outcome_datetime", "outcome_type"},
        "outcomes",
    )
    result = remove_exact_duplicates(df)
    result = parse_datetime_columns(result, ["outcome_datetime", "date_of_birth"])
    result = result.dropna(subset=["animal_id", "animal_type", "outcome_datetime"])
    result = filter_cats_and_dogs(result)
    return result
