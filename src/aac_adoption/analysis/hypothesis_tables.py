"""H1/H3/H5 support tables for thesis discussion."""

from pathlib import Path

import pandas as pd


def _summarize(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        df.groupby(column, dropna=False)
        .agg(
            records=("adopted", "count"),
            adoptions=("adopted", "sum"),
            adoption_rate_pct=("adopted", lambda values: float(values.mean() * 100)),
            median_days_to_outcome=("days_to_outcome", "median"),
        )
        .reset_index()
        .rename(columns={column: "value"})
        .assign(variable=column)
        .sort_values(["variable", "records"], ascending=[True, False])
    )


def _importance_for(tables_dir: Path, feature_terms: list[str]) -> pd.DataFrame:
    paths = [
        tables_dir / "permutation_importance_classification.csv",
        tables_dir / "random_forest_feature_importance.csv",
        tables_dir / "logistic_regression_coefficients.csv",
    ]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame(columns=["feature", "importance_score"])

    normalized = []
    for frame in frames:
        score_col = None
        for candidate in ["importance_mean", "importance", "abs_coefficient"]:
            if candidate in frame.columns:
                score_col = candidate
                break
        if score_col is None or "feature" not in frame.columns:
            continue
        normalized.append(frame[["feature", score_col]].rename(columns={score_col: "importance_score"}))
    if not normalized:
        return pd.DataFrame(columns=["feature", "importance_score"])

    importance = pd.concat(normalized, ignore_index=True)
    mask = importance["feature"].astype(str).str.contains("|".join(feature_terms), case=False, regex=True)
    return (
        importance.loc[mask]
        .groupby("feature", as_index=False)["importance_score"]
        .mean()
        .sort_values("importance_score", ascending=False)
    )


def _write_with_importance(
    summary: pd.DataFrame,
    importance: pd.DataFrame,
    output_path: Path,
) -> None:
    if importance.empty:
        summary["related_importance_features"] = ""
        summary["mean_importance_score"] = pd.NA
    else:
        summary["related_importance_features"] = "; ".join(importance["feature"].head(10).astype(str))
        summary["mean_importance_score"] = importance["importance_score"].mean()
    summary.to_csv(output_path, index=False)


def create_hypothesis_support_tables(
    data_path: str | Path = "data/processed/modeling_dataset.csv",
    tables_dir: str | Path = "reports/tables",
) -> None:
    """Create thesis support tables for central hypotheses H1, H3, and H5."""
    tables = Path(tables_dir)
    tables.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)

    h1 = pd.concat(
        [
            _summarize(df, "intake_type"),
            _summarize(df, "intake_condition"),
            _summarize(df, "simplified_breed_group"),
            _summarize(df, "simplified_color_group"),
        ],
        ignore_index=True,
    )
    _write_with_importance(
        h1,
        _importance_for(
            tables,
            ["intake_type", "intake_condition", "simplified_breed_group", "simplified_color_group"],
        ),
        tables / "h1_intake_vs_appearance.csv",
    )

    h3 = _summarize(df, "age_group")
    _write_with_importance(
        h3,
        _importance_for(tables, ["age_days", "age_months", "age_years", "age_group"]),
        tables / "h3_age_adoption_speed.csv",
    )

    h5 = _summarize(df, "covid_period")
    _write_with_importance(
        h5,
        _importance_for(tables, ["covid_period"]),
        tables / "h5_covid_period.csv",
    )

