"""Animal-centered profile and journey analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROFILE_COLUMNS = [
    "animal_type",
    "age_group",
    "intake_type",
    "intake_condition",
    "health_profile",
    "behavior_support_flag",
    "simplified_breed_group",
    "simplified_color_group",
    "sex_upon_intake",
    "is_named",
]

URGENT_HEALTH = {"Agonal", "Med Urgent", "Neurologic", "Parvo", "Panleuk", "Congenital"}
MEDICAL_HEALTH = {"Sick", "Injured", "Medical", "Med Attn"}
LIFE_STAGE_HEALTH = {"Nursing", "Neonatal", "Aged", "Pregnant"}
BEHAVIOR_HEALTH = {"Behavior", "Feral"}
BEHAVIOR_SUBTYPES = {"Aggressive", "Behavior", "Rabies Risk", "Court/Investigation"}


def add_animal_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Add EDA-only health and behavior-support descriptors."""
    result = df.copy()
    condition = result.get("intake_condition", pd.Series(index=result.index, dtype="object"))
    subtype = result.get("outcome_subtype", pd.Series(index=result.index, dtype="object"))
    result["health_profile"] = np.select(
        [
            condition.isin(["Normal"]),
            condition.isin(MEDICAL_HEALTH),
            condition.isin(LIFE_STAGE_HEALTH),
            condition.isin(BEHAVIOR_HEALTH),
            condition.isin(URGENT_HEALTH),
        ],
        ["normal", "medical_or_injured", "vulnerable_life_stage", "behavior_or_feral", "urgent_medical"],
        default="other_unknown",
    )
    result["behavior_support_flag"] = np.where(
        condition.isin(BEHAVIOR_HEALTH) | subtype.isin(BEHAVIOR_SUBTYPES),
        "behavior_support_signal",
        "no_behavior_signal",
    )
    return result


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_barh(df: pd.DataFrame, label: str, value: str, path: Path, title: str, xlabel: str, limit: int = 15) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df.head(limit).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df[label].astype(str), plot_df[value].astype(float))
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _profile_label(row: pd.Series) -> str:
    name_status = "has recorded name" if bool(row.get("is_named", False)) else "no recorded name"
    condition = row["health_profile"] if "health_profile" in row else row.get("intake_condition", "unknown")
    sex = row.get("sex_upon_intake", "unknown sex")
    return (
        f"{row.get('age_group', 'unknown')} {row.get('animal_type', 'unknown')} | "
        f"{name_status} / {sex} | "
        f"{row.get('intake_type', 'unknown')} / {condition} | "
        f"{row.get('simplified_breed_group', 'unknown')} / {row.get('simplified_color_group', 'unknown')}"
    )


def _visibility_label(row: pd.Series) -> str:
    adoption = row["adoption_rate_pct"]
    median_days = row["median_days_to_outcome"]
    if adoption >= 60 and median_days < 14:
        return "quick placement likely"
    if adoption >= 50 and median_days >= 14:
        return "needs visibility"
    if adoption < 40 and median_days >= 14:
        return "long-stay risk"
    return "outcome support priority"


def animal_archetypes(df: pd.DataFrame, min_records: int = 50) -> pd.DataFrame:
    """Create animal profile archetype table."""
    available = [column for column in PROFILE_COLUMNS if column in df.columns]
    grouped = (
        df.groupby(available, dropna=False)
        .agg(
            records=("classification_target", "count"),
            adoptions=("classification_target", "sum"),
            adoption_rate_pct=("classification_target", lambda x: float(x.mean() * 100)),
            median_days_to_outcome=("days_to_outcome", "median"),
            mean_days_to_outcome=("days_to_outcome", "mean"),
        )
        .reset_index()
    )
    for outcome in ["Transfer", "Return to Owner", "Euthanasia"]:
        grouped[f"{outcome.lower().replace(' ', '_')}_rate_pct"] = grouped.apply(
            lambda row, outcome=outcome: 0.0,
            axis=1,
        )
    if "outcome_type" in df.columns:
        outcome_mix = (
            df.groupby(available + ["outcome_type"], dropna=False)
            .size()
            .reset_index(name="outcome_records")
        )
        totals = df.groupby(available, dropna=False).size().reset_index(name="total_records")
        outcome_mix = outcome_mix.merge(totals, on=available, how="left")
        outcome_mix["outcome_rate_pct"] = outcome_mix["outcome_records"] / outcome_mix["total_records"] * 100
        pivot = outcome_mix.pivot_table(
            index=available,
            columns="outcome_type",
            values="outcome_rate_pct",
            fill_value=0,
        ).reset_index()
        grouped = grouped.drop(
            columns=[column for column in grouped.columns if column.endswith("_rate_pct") and column != "adoption_rate_pct"],
            errors="ignore",
        ).merge(pivot, on=available, how="left")
        grouped = grouped.rename(
            columns={
                "Adoption": "outcome_adoption_rate_pct",
                "Transfer": "transfer_rate_pct",
                "Return to Owner": "return_to_owner_rate_pct",
                "Euthanasia": "euthanasia_rate_pct",
            }
        )
    for column in ["transfer_rate_pct", "return_to_owner_rate_pct", "euthanasia_rate_pct"]:
        if column not in grouped.columns:
            grouped[column] = 0.0
    grouped["profile_label"] = grouped.apply(_profile_label, axis=1)
    grouped["visibility_need"] = grouped.apply(_visibility_label, axis=1)
    grouped = grouped[grouped["records"] >= min_records].sort_values(["records", "adoption_rate_pct"], ascending=[False, False])
    return grouped


def vulnerable_profiles(archetypes: pd.DataFrame) -> pd.DataFrame:
    """Rank animal profiles that may need visibility or support."""
    result = archetypes.copy()
    result["vulnerability_score"] = (
        (100 - result["adoption_rate_pct"])
        + np.log1p(result["median_days_to_outcome"].clip(lower=0)) * 10
        + np.log1p(result["records"])
    )
    return result.sort_values(["vulnerability_score", "records"], ascending=[False, False]).head(30)


def profile_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """Create key animal-centered contrast table."""
    rows = []

    dogs = df[df["animal_type"].astype(str).str.lower().eq("dog")].copy()
    if not dogs.empty:
        dogs["contrast_value"] = np.where(dogs["simplified_breed_group"].eq("pit_bull_type"), "pit_bull_type_dogs", "other_dogs")
        rows.append(_contrast_summary(dogs, "pit_bull_type_vs_other_dogs"))

    cats = df[df["animal_type"].astype(str).str.lower().eq("cat")].copy()
    if not cats.empty:
        cats["contrast_value"] = np.where(cats["simplified_color_group"].eq("black_or_dark"), "black_or_dark_cats", "other_cats")
        rows.append(_contrast_summary(cats, "black_or_dark_cats_vs_other_cats"))
        cats["contrast_value"] = np.where(cats["simplified_breed_group"].eq("domestic_cat"), "domestic_cat_group", "other_cat_breed_groups")
        rows.append(_contrast_summary(cats, "domestic_cat_vs_other_cat_groups"))

    if "age_group" in df.columns:
        age = df[df["age_group"].isin(["baby", "senior"])].copy()
        age["contrast_value"] = age["animal_type"].astype(str).str.lower() + "_" + age["age_group"].astype(str)
        rows.append(_contrast_summary(age, "senior_vs_baby_by_species"))

    named = df.copy()
    named["contrast_value"] = named["animal_type"].astype(str).str.lower() + "_" + np.where(named["is_named"].astype(bool), "named", "unnamed")
    rows.append(_contrast_summary(named, "named_vs_unnamed_by_species"))

    health = df.copy()
    health["contrast_value"] = health["animal_type"].astype(str).str.lower() + "_" + health["health_profile"].astype(str)
    rows.append(_contrast_summary(health, "health_profile_by_species"))

    behavior = df.copy()
    behavior["contrast_value"] = behavior["animal_type"].astype(str).str.lower() + "_" + behavior["behavior_support_flag"].astype(str)
    rows.append(_contrast_summary(behavior, "behavior_support_signal_by_species"))

    return pd.concat(rows, ignore_index=True)


def _contrast_summary(df: pd.DataFrame, contrast: str) -> pd.DataFrame:
    table = (
        df.groupby("contrast_value", dropna=False)
        .agg(
            records=("classification_target", "count"),
            adoption_rate_pct=("classification_target", lambda x: float(x.mean() * 100)),
            median_days_to_outcome=("days_to_outcome", "median"),
            euthanasia_rate_pct=("outcome_type", lambda x: float((x == "Euthanasia").mean() * 100)),
            transfer_rate_pct=("outcome_type", lambda x: float((x == "Transfer").mean() * 100)),
        )
        .reset_index()
    )
    table["contrast"] = contrast
    return table


def profile_model_error(archetypes: pd.DataFrame, diagnostics_predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Summarize model behavior by profile when diagnostic predictions exist."""
    if diagnostics_predictions is None or diagnostics_predictions.empty:
        return pd.DataFrame()
    available = [column for column in PROFILE_COLUMNS if column in diagnostics_predictions.columns]
    table = (
        diagnostics_predictions.groupby(available, dropna=False)
        .agg(
            records=("classification_target", "count"),
            adoption_rate_pct=("classification_target", lambda x: float(x.mean() * 100)),
            mean_predicted_adoption_probability=("predicted_adoption_probability", lambda x: float(x.mean() * 100)),
            median_predicted_days=("predicted_days_to_outcome", "median"),
            mae=("absolute_error", "mean"),
        )
        .reset_index()
    )
    if table.empty:
        return table
    table["prediction_gap_pct"] = table["mean_predicted_adoption_probability"] - table["adoption_rate_pct"]
    table["profile_label"] = table.apply(_profile_label, axis=1)
    return table.merge(archetypes[["profile_label", "visibility_need"]], on="profile_label", how="left")


def create_animal_profile_outputs(
    data_path: str | Path,
    diagnostics_dir: str | Path = "reports/diagnostics",
    tables_dir: str | Path = "reports/tables",
    figures_dir: str | Path = "reports/figures",
    min_records: int = 50,
) -> None:
    """Generate animal-centered exploratory outputs."""
    df = pd.read_csv(data_path)
    df = add_animal_descriptors(df)
    tables = Path(tables_dir)
    figures = Path(figures_dir)
    diagnostics_path = Path(diagnostics_dir) / "diagnostic_predictions_sample.csv"
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.exists() else pd.DataFrame()
    if not diagnostics.empty:
        diagnostics = add_animal_descriptors(diagnostics)

    archetypes = animal_archetypes(df, min_records=min_records)
    vulnerable = vulnerable_profiles(archetypes)
    contrasts = profile_contrasts(df)
    model_error = profile_model_error(archetypes, diagnostics)

    _save_table(archetypes, tables / "animal_archetypes.csv")
    _save_table(vulnerable, tables / "vulnerable_profiles.csv")
    _save_table(contrasts, tables / "profile_contrasts.csv")
    _save_table(model_error, tables / "profile_model_error.csv")
    _save_table(
        contrasts[contrasts["contrast"].isin(["health_profile_by_species", "behavior_support_signal_by_species"])],
        tables / "health_behavior_profiles.csv",
    )

    _save_barh(archetypes, "profile_label", "records", figures / "animal_archetypes_top.png", "Largest animal archetypes", "Records")
    _save_barh(vulnerable, "profile_label", "vulnerability_score", figures / "vulnerable_profiles.png", "Profiles needing visibility or support", "Vulnerability score")
    _save_barh(contrasts, "contrast_value", "adoption_rate_pct", figures / "profile_contrasts_adoption_rate.png", "Key animal profile contrasts", "Adoption rate (%)")
