"""Raw AAC CSV loading utilities."""

from pathlib import Path
import re

import pandas as pd


def to_snake_case(name: str) -> str:
    """Convert AAC column names such as 'Animal ID' to 'animal_id'."""
    cleaned = name.strip().replace("/", " ")
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned.lower()


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return copy with snake_case column names."""
    result = df.copy()
    result.columns = [to_snake_case(column) for column in result.columns]
    return result


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load CSV and standardize columns without changing raw file."""
    return standardize_column_names(pd.read_csv(path))


def load_intakes(path: str | Path) -> pd.DataFrame:
    """Load AAC intake data and normalize intake datetime naming."""
    df = load_csv(path)
    if "datetime" in df.columns and "intake_datetime" not in df.columns:
        df = df.rename(columns={"datetime": "intake_datetime"})
    return df


def load_outcomes(path: str | Path) -> pd.DataFrame:
    """Load AAC outcome data and normalize outcome datetime naming."""
    df = load_csv(path)
    if "datetime" in df.columns and "outcome_datetime" not in df.columns:
        df = df.rename(columns={"datetime": "outcome_datetime"})
    return df

