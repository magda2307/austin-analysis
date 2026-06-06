"""Feature engineering for AAC adoption modeling."""

import re

import numpy as np
import pandas as pd


AGE_UNIT_TO_DAYS = {
    "day": 1,
    "days": 1,
    "week": 7,
    "weeks": 7,
    "month": 30.4375,
    "months": 30.4375,
    "year": 365.25,
    "years": 365.25,
}

AREA_IN_PATTERN = re.compile(r"\bin\s+(?P<area>[^()]+?)\s*\(tx\)\s*$", re.IGNORECASE)
AREA_EXACT_PATTERN = re.compile(r"^(?P<area>[^()]+?)\s*\(tx\)\s*$", re.IGNORECASE)
INTERSECTION_PATTERN = re.compile(r"\bintersection\b|/|&|\band\b", re.IGNORECASE)
AIRPORT_PATTERN = re.compile(r"\bairport\b|presidential\s+blvd", re.IGNORECASE)


def parse_age_to_days(value: object) -> float:
    """Parse AAC age strings like '2 years' or '7 months' to days."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    match = re.match(r"^(-?\d+(?:\.\d+)?)\s+([a-z]+)", text)
    if not match:
        return np.nan

    amount = float(match.group(1))
    unit = match.group(2)
    if amount < 0 or unit not in AGE_UNIT_TO_DAYS:
        return np.nan

    return amount * AGE_UNIT_TO_DAYS[unit]


def winsorize_outliers(series: pd.Series, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.Series:
    """Winsorize extreme outliers using quantile-based capping."""
    lower = series.quantile(lower_quantile)
    upper = series.quantile(upper_quantile)
    return series.clip(lower=lower, upper=upper)


def season_from_month(month: int) -> str:
    """Map month number to meteorological season."""
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def age_group_from_days(age_days: float) -> str:
    """Create broad age groups useful for thesis EDA and modeling."""
    if pd.isna(age_days):
        return "unknown"
    if age_days < 365.25:
        return "baby"
    if age_days < 3 * 365.25:
        return "young"
    if age_days < 8 * 365.25:
        return "adult"
    return "senior"


def covid_period_from_date(value: pd.Timestamp) -> str:
    """Label intake timing for later COVID-period analysis."""
    if pd.isna(value):
        return "unknown"
    if value < pd.Timestamp("2020-03-01"):
        return "pre_covid"
    if value < pd.Timestamp("2022-01-01"):
        return "covid"
    return "post_covid"


def simplify_color(value: object) -> str:
    """Create coarse color groups, including dark/black flag support."""
    if pd.isna(value):
        return "unknown"

    text = str(value).strip().lower()
    if not text:
        return "unknown"
    if any(token in text for token in ["black", "sable"]):
        return "black_or_dark"
    if any(token in text for token in ["brown", "chocolate", "liver", "seal"]):
        return "brown_tan"
    if any(token in text for token in ["white", "cream"]):
        return "white_light"
    if any(token in text for token in ["gray", "grey", "blue", "silver"]):
        return "gray_blue"
    if any(token in text for token in ["orange", "yellow", "gold", "tan", "buff", "fawn"]):
        return "orange_yellow"
    if any(token in text for token in ["calico", "torbie", "tortie", "tricolor", "brindle"]):
        return "mixed_other"
    return "mixed_other"


def primary_color(value: object) -> str:
    """Return first AAC color token."""
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    return text.replace(" ", "_").split("/")[0]


def primary_breed(value: object) -> str:
    """Return first AAC breed token before mix/slash markers."""
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    text = text.replace(" mix", "")
    return text.split("/")[0].strip().replace(" ", "_")


def is_mixed_breed(value: object) -> bool:
    """Detect obvious mixed-breed labels."""
    if pd.isna(value):
        return False
    text = str(value).lower()
    return "mix" in text or "/" in text


def simplified_breed_group(value: object) -> str:
    """Coarse breed group for early models and descriptive analysis."""
    breed = primary_breed(value)
    if breed == "unknown":
        return "unknown"
    if any(token in breed for token in ["domestic_shorthair", "domestic_medium_hair", "domestic_longhair"]):
        return "domestic_cat"
    if any(token in breed for token in ["pit_bull", "staffordshire", "bulldog"]):
        return "pit_bull_type"
    if any(token in breed for token in ["chihuahua"]):
        return "chihuahua_type"
    if any(token in breed for token in ["retriever", "labrador", "golden"]):
        return "retriever_type"
    if any(token in breed for token in ["shepherd"]):
        return "shepherd_type"
    if any(token in breed for token in ["terrier"]):
        return "terrier_type"
    if any(token in breed for token in ["hound"]):
        return "hound_type"
    return "other"


def normalize_location_area(value: str) -> str:
    """Normalize extracted location labels into stable categorical tokens."""
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unknown"


def extract_found_location_area(value: object) -> str:
    """Extract trailing AAC area labels such as Austin or Travis from Found Location."""
    if pd.isna(value):
        return "unknown"

    text = str(value).strip()
    if not text:
        return "unknown"

    match = AREA_IN_PATTERN.search(text) or AREA_EXACT_PATTERN.search(text)
    if not match:
        return "unknown"
    return normalize_location_area(match.group("area"))


def found_location_flags(value: object) -> dict[str, bool]:
    """Return deterministic intake-time flags for AAC Found Location text."""
    if pd.isna(value):
        text = ""
    else:
        text = str(value).strip()
    lower = text.lower()

    return {
        "is_austin_found_location": extract_found_location_area(text) == "austin",
        "is_outside_jurisdiction": lower == "outside jurisdiction",
        "is_intersection_location": bool(INTERSECTION_PATTERN.search(text)),
        "is_address_like_location": bool(re.match(r"^\s*\d+", text)),
        "is_airport_location": bool(AIRPORT_PATTERN.search(text)),
    }


def found_location_kind(value: object) -> str:
    """Classify AAC Found Location into a coarse, leakage-safe taxonomy."""
    if pd.isna(value):
        return "other"

    text = str(value).strip()
    if not text:
        return "other"

    flags = found_location_flags(text)
    area = extract_found_location_area(text)

    if flags["is_outside_jurisdiction"]:
        return "outside_jurisdiction"
    if flags["is_intersection_location"]:
        return "intersection"
    if flags["is_address_like_location"]:
        return "address_like"
    if area == "austin":
        return "austin_city"
    if area != "unknown":
        return "county_or_region"
    return "other"


def add_intake_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add intake-time features only."""
    result = df.copy()

    result["has_name"] = (
        result.get("name", pd.Series(index=result.index, dtype="object"))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    result["is_named"] = result["has_name"]

    result["age_in_days"] = result.get("age_upon_intake", pd.Series(index=result.index)).map(
        parse_age_to_days
    )
    result["age_in_months"] = result["age_in_days"] / 30.4375
    result["age_in_years"] = result["age_in_days"] / 365.25
    result["age_days"] = result["age_in_days"]
    result["age_months"] = result["age_in_months"]
    result["age_years"] = result["age_in_years"]
    result["age_group"] = result["age_in_days"].map(age_group_from_days)

    result["intake_year"] = result["intake_datetime"].dt.year
    result["intake_month"] = result["intake_datetime"].dt.month
    result["intake_quarter"] = result["intake_datetime"].dt.quarter
    result["intake_season"] = result["intake_month"].map(season_from_month)
    result["covid_period"] = result["intake_datetime"].map(covid_period_from_date)

    result["color_group"] = result.get("color", pd.Series(index=result.index)).map(simplify_color)
    result["primary_color"] = result.get("color", pd.Series(index=result.index)).map(primary_color)
    result["simplified_color_group"] = result["color_group"]
    result["is_black_or_dark"] = result["simplified_color_group"].eq("black_or_dark")
    result["primary_breed"] = result.get("breed", pd.Series(index=result.index)).map(primary_breed)
    result["is_mixed_breed"] = result.get("breed", pd.Series(index=result.index)).map(is_mixed_breed)
    result["simplified_breed_group"] = result.get("breed", pd.Series(index=result.index)).map(
        simplified_breed_group
    )

    found_location = result.get("found_location", pd.Series(index=result.index))
    flags = found_location.map(found_location_flags)
    result["found_location_kind"] = found_location.map(found_location_kind)
    result["found_location_area"] = found_location.map(extract_found_location_area)
    for column in [
        "is_austin_found_location",
        "is_outside_jurisdiction",
        "is_intersection_location",
        "is_address_like_location",
        "is_airport_location",
    ]:
        result[column] = flags.map(lambda values, name=column: values[name])
    return result
