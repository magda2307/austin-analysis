"""Feature family mapping for thesis-level interpretation."""

from __future__ import annotations


FEATURE_FAMILY_TERMS = {
    "age": ["age_", "age_days", "age_months", "age_years", "age_group", "age_upon_intake"],
    "intake_circumstances": ["intake_type"],
    "intake_condition_health": ["intake_condition"],
    "sex_reproductive_status": ["sex_upon_intake"],
    "breed_appearance": ["breed", "primary_breed", "simplified_breed_group", "is_mixed_breed"],
    "color": ["color", "primary_color", "simplified_color_group", "is_black_or_dark"],
    "name_identity": ["has_name", "is_named"],
    "timing_seasonality": ["intake_year", "intake_month", "intake_quarter", "intake_season"],
    "covid_period": ["covid_period"],
    "weather_context": ["daily_temp", "daily_precipitation", "is_extreme_heat", "is_rainy_day"],
    "shelter_demand_context": ["animal_311_requests", "intake_volume"],
    "animal_type": ["animal_type"],
}


def feature_family(feature: str) -> str:
    """Map a feature name to a thesis-friendly family."""
    feature_lower = feature.lower()
    for family, terms in FEATURE_FAMILY_TERMS.items():
        if any(term in feature_lower for term in terms):
            return family
    return "other"

