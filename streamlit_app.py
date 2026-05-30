"""Streamlit thesis demo for AAC adoption analysis."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aac_adoption.dashboard.data import (  # noqa: E402
    best_model_rows,
    build_prediction_record,
    load_summary,
    load_table,
    predict_from_record,
)


TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SUMMARY_DIR = PROJECT_ROOT / "reports" / "summary"
MODELS_DIR = PROJECT_ROOT / "models" / "boosting"


st.set_page_config(
    page_title="AAC Adoption Thesis Demo",
    page_icon="",
    layout="wide",
)


@st.cache_data
def cached_tables() -> dict[str, pd.DataFrame]:
    return {
        "classification": load_table(TABLES_DIR, "classification"),
        "regression": load_table(TABLES_DIR, "regression"),
        "h1": load_table(TABLES_DIR, "h1"),
        "h3": load_table(TABLES_DIR, "h3"),
        "h5": load_table(TABLES_DIR, "h5"),
    }


def show_metric_cards(best_rows: pd.DataFrame) -> None:
    if best_rows.empty:
        st.info("Run the training and analysis pipeline to populate model comparison outputs.")
        return

    ordered = best_rows.sort_values(["task", "animal_subset"])
    columns = st.columns(min(len(ordered), 6))
    for column, (_, row) in zip(columns, ordered.iterrows()):
        metric_label = row["primary_metric"].upper()
        value = f"{row['score']:.3f}" if row["primary_metric"] != "mae" else f"{row['score']:.2f} days"
        column.metric(
            f"{row['animal_subset']} {row['task']}",
            value,
            help=f"{row['model_name']} by {metric_label}",
        )


def figure(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Missing figure: {path.name}")


tables = cached_tables()
best_rows = best_model_rows(tables["classification"], tables["regression"])

st.title("AAC Adoption Thesis Demo")
st.caption("Artifact-driven dashboard for model results, hypothesis signals, and what-if predictions.")

tabs = st.tabs(["Overview", "Models", "Hypotheses", "What-if Prediction", "Artifacts"])

with tabs[0]:
    show_metric_cards(best_rows)
    st.markdown(load_summary(SUMMARY_DIR))

with tabs[1]:
    left, right = st.columns(2)
    with left:
        figure(FIGURES_DIR / "model_comparison_classification_roc_auc.png", "Classification ROC-AUC")
        figure(FIGURES_DIR / "model_comparison_classification_f1.png", "Classification F1")
    with right:
        figure(FIGURES_DIR / "model_comparison_regression_mae.png", "Regression MAE")
        figure(FIGURES_DIR / "model_comparison_regression_rmse.png", "Regression RMSE")

    st.subheader("Classification Table")
    st.dataframe(tables["classification"], use_container_width=True, hide_index=True)
    st.subheader("Regression Table")
    st.dataframe(tables["regression"], use_container_width=True, hide_index=True)

with tabs[2]:
    h1_left, h1_right = st.columns(2)
    with h1_left:
        figure(FIGURES_DIR / "h1_intake_type_adoption_rate.png", "H1 adoption rate by intake type")
        figure(FIGURES_DIR / "h3_age_group_adoption_rate.png", "H3 adoption rate by age group")
        figure(FIGURES_DIR / "h5_covid_period_adoption_rate.png", "H5 adoption rate by COVID period")
    with h1_right:
        figure(FIGURES_DIR / "h1_intake_condition_adoption_rate.png", "H1 adoption rate by intake condition")
        figure(FIGURES_DIR / "h3_age_group_median_days.png", "H3 median days by age group")
        figure(FIGURES_DIR / "h5_covid_period_median_days.png", "H5 median days by COVID period")

    st.subheader("H1: Intake vs Appearance")
    st.dataframe(tables["h1"], use_container_width=True, hide_index=True)
    st.subheader("H3: Age and Adoption Speed")
    st.dataframe(tables["h3"], use_container_width=True, hide_index=True)
    st.subheader("H5: COVID-period Dynamics")
    st.dataframe(tables["h5"], use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("What-if Prediction")
    st.caption("Uses the combined histogram gradient boosting classifier and regressor. This is a demo prediction, not a causal decision rule.")

    left, right = st.columns(2)
    with left:
        animal_type = st.selectbox("Animal type", ["Dog", "Cat"])
        intake_type = st.selectbox(
            "Intake type",
            ["Stray", "Owner Surrender", "Public Assist", "Abandoned", "Euthanasia Request"],
        )
        intake_condition = st.selectbox(
            "Intake condition",
            ["Normal", "Injured", "Sick", "Nursing", "Neonatal", "Aged", "Medical", "Behavior", "Other"],
        )
        sex_upon_intake = st.selectbox(
            "Sex upon intake",
            ["Intact Male", "Intact Female", "Neutered Male", "Spayed Female", "Unknown"],
        )
        has_name = st.toggle("Has name", value=True)
    with right:
        age_years = st.slider("Age in years", min_value=0.0, max_value=20.0, value=2.0, step=0.25)
        breed = st.text_input("Breed", value="Labrador Retriever Mix" if animal_type == "Dog" else "Domestic Shorthair Mix")
        color = st.text_input("Color", value="Black/White")
        intake_date = st.date_input("Intake date", value=date(2024, 6, 1))

    record = build_prediction_record(
        animal_type=animal_type,
        intake_type=intake_type,
        intake_condition=intake_condition,
        sex_upon_intake=sex_upon_intake,
        age_days=age_years * 365.25,
        breed=breed,
        color=color,
        has_name=has_name,
        intake_date=pd.Timestamp(intake_date),
    )

    if st.button("Run prediction", type="primary"):
        try:
            prediction = predict_from_record(record, MODELS_DIR)
            probability_pct = prediction["adoption_probability"] * 100
            days = prediction["predicted_days_to_outcome"]
            col1, col2 = st.columns(2)
            col1.metric("Predicted adoption probability", f"{probability_pct:.1f}%")
            col2.metric("Predicted days to outcome", f"{days:.1f}")
            st.dataframe(record, use_container_width=True, hide_index=True)
        except FileNotFoundError as error:
            st.error(str(error))
            st.info("Run `python scripts/train_boosting.py --data data/processed/modeling_dataset.csv` first.")

with tabs[4]:
    st.subheader("Generated Artifacts")
    st.write("Core commands:")
    st.code(
        "\n".join(
            [
                "python scripts/build_dataset.py --intakes data/raw/intakes.csv --outcomes data/raw/outcomes.csv --output data/processed/modeling_dataset.csv",
                "python scripts/run_eda.py --data data/processed/modeling_dataset.csv",
                "python scripts/train_baseline.py --data data/processed/modeling_dataset.csv",
                "python scripts/train_boosting.py --data data/processed/modeling_dataset.csv",
                "python scripts/run_analysis.py --data data/processed/modeling_dataset.csv",
                "python scripts/generate_report_outputs.py",
            ]
        ),
        language="bash",
    )
    st.write("Reports directory:", str(PROJECT_ROOT / "reports"))
    st.write("Models directory:", str(MODELS_DIR))
