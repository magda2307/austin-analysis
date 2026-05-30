"""Initial EDA tables and plots for thesis outputs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def adoption_rate(series: pd.Series) -> float:
    """Mean adoption flag as percentage."""
    return float(series.mean() * 100)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_bar_plot(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.bar(df[x].astype(str), df[y])
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def summarize_adoption_los(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Summarize adoption rate and median LOS by one descriptive column."""
    return (
        df.groupby(column, dropna=False)
        .agg(
            records=("adopted", "count"),
            adoptions=("adopted", "sum"),
            adoption_rate_pct=("adopted", adoption_rate),
            median_days_to_outcome=("days_to_outcome", "median"),
        )
        .reset_index()
        .sort_values("records", ascending=False)
    )


def create_eda_outputs(
    data_path: str | Path,
    tables_dir: str | Path = "reports/tables",
    figures_dir: str | Path = "reports/figures",
) -> None:
    """Create first thesis-ready EDA tables and simple figures."""
    header = pd.read_csv(data_path, nrows=0)
    parse_dates = [col for col in ["intake_datetime", "outcome_datetime"] if col in header.columns]
    df = pd.read_csv(data_path, parse_dates=parse_dates)
    tables = Path(tables_dir)
    figures = Path(figures_dir)

    intakes_by_year = (
        df.groupby("intake_year", dropna=False)
        .size()
        .reset_index(name="intakes")
        .sort_values("intake_year")
    )
    save_table(intakes_by_year, tables / "intakes_by_year.csv")
    save_bar_plot(intakes_by_year, "intake_year", "intakes", figures / "intakes_by_year.png", "Intakes by year")

    adoptions_by_year = (
        df.groupby("intake_year", dropna=False)["adopted"]
        .agg(outcomes="count", adoptions="sum", adoption_rate_pct=adoption_rate)
        .reset_index()
        .sort_values("intake_year")
    )
    save_table(adoptions_by_year, tables / "outcomes_adoptions_by_year.csv")
    save_bar_plot(
        adoptions_by_year,
        "intake_year",
        "adoptions",
        figures / "adoptions_by_year.png",
        "Adoptions by intake year",
    )

    adoption_rate_by_type = (
        df.groupby("animal_type")["adopted"]
        .agg(records="count", adoption_rate_pct=adoption_rate)
        .reset_index()
    )
    save_table(adoption_rate_by_type, tables / "adoption_rate_by_animal_type.csv")

    median_los_by_type = (
        df.groupby("animal_type")["days_to_outcome"]
        .median()
        .reset_index(name="median_days_to_outcome")
    )
    save_table(median_los_by_type, tables / "median_los_by_animal_type.csv")

    for column, filename in [
        ("age_group", "adoption_rate_by_age_group.csv"),
        ("intake_type", "adoption_rate_by_intake_type.csv"),
        ("simplified_color_group", "adoption_rate_by_color_group.csv"),
        ("covid_period", "adoption_rate_by_covid_period.csv"),
    ]:
        if column in df.columns:
            table = (
                df.groupby(column, dropna=False)["adopted"]
                .agg(records="count", adoption_rate_pct=adoption_rate)
                .reset_index()
                .sort_values("records", ascending=False)
            )
            save_table(table, tables / filename)

    priority_columns = [
        "intake_type",
        "intake_condition",
        "simplified_breed_group",
        "simplified_color_group",
        "age_group",
        "covid_period",
        "intake_season",
        "is_black_or_dark",
    ]
    for column in priority_columns:
        if column in df.columns:
            save_table(summarize_adoption_los(df, column), tables / f"adoption_los_by_{column}.csv")
