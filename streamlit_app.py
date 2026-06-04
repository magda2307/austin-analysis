"""Streamlit thesis demo for AAC adoption analysis."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import altair as alt

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aac_adoption.dashboard.data import (  # noqa: E402
    best_model_rows,
    build_profile_prediction_record,
    build_prediction_record,
    load_diagnostic,
    load_optional_csv,
    load_summary,
    load_table,
    local_shap_explanations,
    predict_from_record,
    profile_global_shap_reasons,
    similar_historical_cases,
    visibility_need_from_prediction,
)
from aac_adoption.dashboard.story import (  # noqa: E402
    approach_comparison_rows,
    decision_sankey,
    story_cards,
    workflow_dot,
)


TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SUMMARY_DIR = PROJECT_ROOT / "reports" / "summary"
DIAGNOSTICS_DIR = PROJECT_ROOT / "reports" / "diagnostics"
MODELS_DIR = PROJECT_ROOT / "models" / "advanced"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset.csv"


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
        "animal_archetypes": load_table(TABLES_DIR, "animal_archetypes"),
        "vulnerable_profiles": load_table(TABLES_DIR, "vulnerable_profiles"),
        "profile_contrasts": load_table(TABLES_DIR, "profile_contrasts"),
        "profile_model_error": load_table(TABLES_DIR, "profile_model_error"),
        "health_behavior_profiles": load_table(TABLES_DIR, "health_behavior_profiles"),
        "model_evidence_pack": load_table(TABLES_DIR, "model_evidence_pack"),
        "model_limitations_by_cohort": load_table(TABLES_DIR, "model_limitations_by_cohort"),
        "metric_confidence_intervals": load_table(TABLES_DIR, "metric_confidence_intervals"),
        "subgroup_reliability": load_table(TABLES_DIR, "subgroup_reliability"),
        "subgroup_metric_confidence_intervals": load_table(TABLES_DIR, "subgroup_metric_confidence_intervals"),
        "subgroup_adoption_milestones": load_table(TABLES_DIR, "subgroup_adoption_milestones"),
        "model_failure_modes": load_table(TABLES_DIR, "model_failure_modes"),
        "animal_journey_examples": load_table(TABLES_DIR, "animal_journey_examples"),
        "shap_classification": load_optional_csv(TABLES_DIR, "shap_global_classification.csv"),
        "shap_regression": load_optional_csv(TABLES_DIR, "shap_global_regression.csv"),
        "shap_family_classification": load_optional_csv(TABLES_DIR, "shap_feature_families_classification.csv"),
        "shap_family_regression": load_optional_csv(TABLES_DIR, "shap_feature_families_regression.csv"),
        "milestones": load_optional_csv(TABLES_DIR, "adoption_by_day_milestones.csv"),
    }


@st.cache_data
def cached_diagnostics() -> dict[str, pd.DataFrame]:
    return {
        "thresholds": load_diagnostic(DIAGNOSTICS_DIR, "thresholds"),
        "calibration": load_diagnostic(DIAGNOSTICS_DIR, "calibration"),
        "classification_slices": load_diagnostic(DIAGNOSTICS_DIR, "classification_slices"),
        "regression_slices": load_diagnostic(DIAGNOSTICS_DIR, "regression_slices"),
        "risk_quadrants": load_diagnostic(DIAGNOSTICS_DIR, "risk_quadrants"),
        "predictions": load_diagnostic(DIAGNOSTICS_DIR, "predictions"),
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
diagnostics = cached_diagnostics()
best_rows = best_model_rows(tables["classification"], tables["regression"])

st.title("AAC Adoption Thesis Demo")
st.caption("Artifact-driven dashboard for model results, hypothesis signals, and what-if predictions.")

tabs = st.tabs(
    [
        "Executive Overview",
        "Story Mode",
        "Animal Stories",
        "Model Quality",
        "Trust & Limits",
        "Interpretability",
        "Risk Explorer",
        "Hypothesis Lab",
        "Campaign Finder",
        "What-if Prediction",
        "Adoption Timeline",
        "Artifacts",
    ]
)

with tabs[0]:
    show_metric_cards(best_rows)
    st.markdown(load_summary(SUMMARY_DIR))

with tabs[1]:
    st.subheader("Data-to-Decision Story")
    st.caption("How raw shelter records become thesis evidence and practical shelter-facing signals.")
    st.graphviz_chart(workflow_dot(), use_container_width=True)
    st.plotly_chart(decision_sankey(), use_container_width=True)

    st.subheader("Approach Comparison")
    approaches = approach_comparison_rows()
    st.altair_chart(
        alt.Chart(approaches)
        .mark_bar()
        .encode(
            y=alt.Y("layer:N", sort=None, title="Analytical layer"),
            x=alt.X("count():Q", title="Story weight"),
            color=alt.Color("layer:N", legend=None),
            tooltip=["layer", "technology", "answers", "strength", "dashboard_use"],
        )
        .properties(height=280),
        use_container_width=True,
    )
    st.dataframe(approaches, use_container_width=True, hide_index=True)

    st.subheader("Real-life Shelter Questions")
    card_columns = st.columns(5)
    for column, card in zip(card_columns, story_cards()):
        column.metric(card["title"], card["artifact"])
        column.caption(card["question"])

with tabs[2]:
    st.subheader("Animal Journey Cards")
    archetypes = tables["animal_archetypes"]
    if archetypes.empty:
        st.info("Run `python scripts/generate_animal_research.py --data data/processed/modeling_dataset.csv` to populate animal stories.")
    else:
        labels = archetypes["profile_label"].head(250).tolist()
        selected_label = st.selectbox("Animal profile", labels)
        selected = archetypes[archetypes["profile_label"].eq(selected_label)].iloc[0]
        profile_record = build_profile_prediction_record(selected)
        profile_prediction: dict[str, float] | None = None
        profile_similarity = similar_historical_cases(DATA_PATH, profile_record)
        try:
            profile_prediction = predict_from_record(profile_record, MODELS_DIR)
        except FileNotFoundError:
            profile_prediction = None

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Similar records", f"{int(selected['records']):,}")
        col2.metric("Adoption rate", f"{selected['adoption_rate_pct']:.1f}%")
        col3.metric("Median wait", f"{selected['median_days_to_outcome']:.1f} days")
        col4.metric("Visibility need", selected["visibility_need"])

        st.write(
            f"**Profile:** {selected['animal_type']} | {selected['age_group']} | "
            f"{selected['intake_type']} / {selected['intake_condition']} | "
            f"{selected.get('health_profile', 'unknown health')} | "
            f"{selected.get('behavior_support_flag', 'unknown behavior signal')} | "
            f"{selected['simplified_breed_group']} / {selected['simplified_color_group']} | "
            f"{'named' if bool(selected['is_named']) else 'unnamed'}"
        )
        mix_cols = st.columns(3)
        mix_cols[0].metric("Transfer rate", f"{selected.get('transfer_rate_pct', 0):.1f}%")
        mix_cols[1].metric("Return-to-owner rate", f"{selected.get('return_to_owner_rate_pct', 0):.1f}%")
        mix_cols[2].metric("Euthanasia rate", f"{selected.get('euthanasia_rate_pct', 0):.1f}%")

        st.subheader("Model View for This Journey")
        if profile_prediction is None:
            st.info("Run `python scripts/train_advanced.py --data data/processed/modeling_dataset.csv` to add representative CatBoost predictions to journey cards.")
        else:
            predicted_probability = profile_prediction["adoption_probability"]
            predicted_days = profile_prediction["predicted_days_to_outcome"]
            model_cols = st.columns(3)
            model_cols[0].metric("Predicted adoption chance", f"{predicted_probability * 100:.1f}%")
            model_cols[1].metric("Predicted wait", f"{predicted_days:.1f} days")
            model_cols[2].metric("Model visibility label", visibility_need_from_prediction(predicted_probability, predicted_days))
            with st.expander("Representative model record"):
                st.dataframe(profile_record, use_container_width=True, hide_index=True)

        st.subheader("Similar Historical Cases")
        if profile_similarity.empty:
            st.info("No similar historical cases found for this representative card.")
        else:
            st.dataframe(profile_similarity, use_container_width=True, hide_index=True)

        st.subheader("Top SHAP Reasons")
        shap_view = pd.DataFrame()
        if profile_prediction is not None:
            try:
                shap_view = local_shap_explanations(profile_record, MODELS_DIR, task="classification", top_n=8)
            except FileNotFoundError:
                shap_view = pd.DataFrame()
        if shap_view.empty:
            shap_view = profile_global_shap_reasons(selected, tables["shap_classification"], top_n=8)
            if shap_view.empty:
                st.info("Run `python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv --include-shap` to populate SHAP reasons.")
            else:
                st.caption("Model-wide SHAP signals mapped onto this animal profile; associations, not causes.")
                st.dataframe(shap_view, use_container_width=True, hide_index=True)
        else:
            st.caption("Local CatBoost SHAP values for the representative journey record; associations, not causes.")
            st.dataframe(shap_view, use_container_width=True, hide_index=True)

    st.subheader("Key Animal Contrasts")
    contrasts = tables["profile_contrasts"]
    if not contrasts.empty:
        contrast_choice = st.selectbox("Contrast", sorted(contrasts["contrast"].unique()))
        contrast_view = contrasts[contrasts["contrast"].eq(contrast_choice)]
        st.altair_chart(
            alt.Chart(contrast_view)
            .mark_bar()
            .encode(
                x=alt.X("contrast_value:N", title="Animal group"),
                y=alt.Y("adoption_rate_pct:Q", title="Adoption rate (%)"),
                color="contrast_value:N",
                tooltip=["contrast_value", "records", "adoption_rate_pct", "median_days_to_outcome", "euthanasia_rate_pct"],
            )
            .properties(height=320),
            use_container_width=True,
        )
        st.dataframe(contrast_view, use_container_width=True, hide_index=True)
    left_animal, right_animal = st.columns(2)
    with left_animal:
        figure(FIGURES_DIR / "animal_archetypes_top.png", "Largest animal archetypes")
    with right_animal:
        figure(FIGURES_DIR / "vulnerable_profiles.png", "Animal profiles needing visibility or support")
    st.subheader("Vulnerable Profiles")
    st.dataframe(tables["vulnerable_profiles"].head(30), use_container_width=True, hide_index=True)
    st.subheader("Health and Behavior Support Profiles")
    st.dataframe(tables["health_behavior_profiles"], use_container_width=True, hide_index=True)

with tabs[3]:
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

    st.subheader("Probability Trust Meter")
    calibration = diagnostics["calibration"]
    if calibration.empty:
        st.info("Run `python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv` to populate calibration diagnostics.")
    else:
        st.altair_chart(
            alt.Chart(calibration)
            .mark_line(point=True)
            .encode(
                x=alt.X("mean_predicted_probability:Q", title="Mean predicted probability"),
                y=alt.Y("observed_adoption_rate:Q", title="Observed adoption rate"),
                tooltip=["probability_bin", "records", "mean_predicted_probability", "observed_adoption_rate"],
            )
            .properties(height=320),
            use_container_width=True,
        )
    st.subheader("Reliability Figures")
    left_diag, right_diag = st.columns(2)
    with left_diag:
        figure(FIGURES_DIR / "diagnostic_roc_curve.png", "Advanced model ROC curve")
        figure(FIGURES_DIR / "diagnostic_precision_recall_curve.png", "Advanced model precision-recall curve")
    with right_diag:
        figure(FIGURES_DIR / "diagnostic_calibration_curve.png", "Probability calibration")
        figure(FIGURES_DIR / "diagnostic_predicted_vs_actual.png", "Regression predicted vs actual")

with tabs[4]:
    st.subheader("Model Evidence Pack")
    evidence = tables["model_evidence_pack"]
    intervals = tables["metric_confidence_intervals"]
    limitations = tables["model_limitations_by_cohort"]
    subgroup_reliability_table = tables["subgroup_reliability"]
    subgroup_intervals = tables["subgroup_metric_confidence_intervals"]
    subgroup_milestones = tables["subgroup_adoption_milestones"]
    failure_modes = tables["model_failure_modes"]
    journey_examples = tables["animal_journey_examples"]
    evidence_summary = load_summary(SUMMARY_DIR).split("## Model Evidence Pack", maxsplit=1)
    if evidence.empty and intervals.empty and limitations.empty:
        st.info("Run `python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv` to populate trust and limits artifacts.")
    else:
        if len(evidence_summary) > 1:
            st.markdown("## Model Evidence Pack" + evidence_summary[1])
        if not intervals.empty:
            st.subheader("Metric Confidence Intervals")
            st.dataframe(intervals, use_container_width=True, hide_index=True)
            st.altair_chart(
                alt.Chart(intervals)
                .mark_rule(size=4)
                .encode(
                    y=alt.Y("metric:N", sort=None, title="Metric"),
                    x=alt.X("lower:Q", title="Bootstrap interval"),
                    x2="upper:Q",
                    color="animal_subset:N",
                    tooltip=["metric", "animal_subset", "lower", "estimate", "upper", "bootstrap_samples"],
                )
                .properties(height=260),
                use_container_width=True,
            )
        if not limitations.empty:
            st.subheader("Cohort Reliability Limits")
            reliable = limitations[~limitations["small_cohort_flag"].astype(bool)] if "small_cohort_flag" in limitations.columns else limitations
            st.dataframe(reliable.head(30), use_container_width=True, hide_index=True)
            st.altair_chart(
                alt.Chart(reliable.head(30))
                .mark_bar()
                .encode(
                    x=alt.X("calibration_gap:Q", title="Calibration gap"),
                    y=alt.Y("value:N", sort="-x", title="Cohort value"),
                    color="cohort:N",
                    tooltip=["cohort", "value", "records", "calibration_gap", "mae", "false_negative_rate"],
                )
                .properties(height=360),
                use_container_width=True,
            )
        if not subgroup_reliability_table.empty:
            st.subheader("Subgroup Explorer")
            subgroup_options = sorted(subgroup_reliability_table["cohort"].dropna().astype(str).unique().tolist())
            subgroup_choice = st.selectbox("Reliability subgroup", subgroup_options)
            subgroup_view = subgroup_reliability_table[subgroup_reliability_table["cohort"].astype(str).eq(subgroup_choice)]
            stable_view = subgroup_view[~subgroup_view["small_cohort_flag"].astype(bool)] if "small_cohort_flag" in subgroup_view.columns else subgroup_view
            st.dataframe(stable_view, use_container_width=True, hide_index=True)
            st.altair_chart(
                alt.Chart(stable_view)
                .mark_bar()
                .encode(
                    x=alt.X("calibration_gap:Q", title="Calibration gap"),
                    y=alt.Y("value:N", sort="-x", title=subgroup_choice.replace("_", " ")),
                    tooltip=["value", "records", "observed_adoption_rate", "mean_predicted_adoption_probability", "calibration_gap", "mae"],
                )
                .properties(height=320),
                use_container_width=True,
            )
        if not failure_modes.empty:
            st.subheader("Where the Model Struggles")
            st.dataframe(failure_modes.head(40), use_container_width=True, hide_index=True)
        if not subgroup_intervals.empty:
            st.subheader("Subgroup Metric Intervals")
            interval_view = subgroup_intervals[subgroup_intervals["status"].eq("ok")] if "status" in subgroup_intervals.columns else subgroup_intervals
            st.dataframe(interval_view.head(50), use_container_width=True, hide_index=True)
        if not subgroup_milestones.empty:
            st.subheader("Time-to-Adoption Milestones")
            milestone_group = st.selectbox("Milestone subgroup", sorted(subgroup_milestones["cohort"].dropna().astype(str).unique().tolist()))
            milestone_view = subgroup_milestones[subgroup_milestones["cohort"].astype(str).eq(milestone_group)].head(20)
            milestone_chart = milestone_view.melt(
                id_vars=["value", "records", "adoptions", "adoption_rate_pct"],
                value_vars=["adopted_by_day_7_pct", "adopted_by_day_30_pct", "adopted_by_day_60_pct", "adopted_by_day_90_pct"],
                var_name="milestone",
                value_name="share",
            )
            st.altair_chart(
                alt.Chart(milestone_chart)
                .mark_bar()
                .encode(
                    x=alt.X("value:N", sort="-y", title=milestone_group.replace("_", " ")),
                    y=alt.Y("share:Q", title="Adopted animals (%)"),
                    color=alt.Color("milestone:N", title="Milestone"),
                    tooltip=[
                        alt.Tooltip("value:N", title="Value"),
                        alt.Tooltip("records:Q", title="Records"),
                        alt.Tooltip("adoptions:Q", title="Adoptions"),
                        alt.Tooltip("adoption_rate_pct:Q", title="Adoption rate"),
                        alt.Tooltip("milestone:N", title="Milestone"),
                        alt.Tooltip("share:Q", title="Adopted by day"),
                    ],
                )
                .properties(height=340),
                use_container_width=True,
            )
            st.dataframe(milestone_view, use_container_width=True, hide_index=True)
        if not journey_examples.empty:
            st.subheader("Animal Journey Evidence Examples")
            st.dataframe(journey_examples, use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader("SHAP Global Explanations")
    st.caption("SHAP values describe factors associated with model predictions, not causal effects.")
    left_shap, right_shap = st.columns(2)
    with left_shap:
        figure(FIGURES_DIR / "shap_summary_classification.png", "Classification SHAP summary")
        st.dataframe(tables["shap_classification"].head(20), use_container_width=True, hide_index=True)
    with right_shap:
        figure(FIGURES_DIR / "shap_summary_regression.png", "Regression SHAP summary")
        st.dataframe(tables["shap_regression"].head(20), use_container_width=True, hide_index=True)
    st.subheader("Feature Family Scores")
    family = tables["shap_family_classification"]
    if not family.empty:
        st.altair_chart(
            alt.Chart(family)
            .mark_bar()
            .encode(
                x=alt.X("mean_abs_shap:Q", title="Sum mean absolute SHAP"),
                y=alt.Y("feature_family:N", sort="-x", title="Feature family"),
                tooltip=["feature_family", "mean_abs_shap", "features"],
            )
            .properties(height=340),
            use_container_width=True,
        )
    else:
        st.info("Run diagnostics with `--include-shap` to populate interpretation artifacts.")

with tabs[6]:
    st.subheader("Risk Threshold Simulator")
    thresholds = diagnostics["thresholds"]
    if thresholds.empty:
        st.info("Run diagnostics to populate threshold tradeoffs.")
    else:
        selected_threshold = st.slider("Adoption probability threshold", 0.05, 0.95, 0.50, 0.05)
        selected = thresholds.iloc[(thresholds["threshold"] - selected_threshold).abs().argsort()[:1]]
        if not selected.empty:
            row = selected.iloc[0]
            cols = st.columns(4)
            cols[0].metric("Precision", f"{row['precision']:.3f}")
            cols[1].metric("Recall", f"{row['recall']:.3f}")
            cols[2].metric("F1", f"{row['f1']:.3f}")
            cols[3].metric("Flagged share", f"{row['flagged_for_adoption_share']:.1%}")
        threshold_chart = thresholds.melt(
            id_vars=["threshold"],
            value_vars=["precision", "recall", "f1"],
            var_name="metric",
            value_name="value",
        )
        st.altair_chart(
            alt.Chart(threshold_chart)
            .mark_line(point=True)
            .encode(
                x=alt.X("threshold:Q", title="Threshold"),
                y=alt.Y("value:Q", title="Metric value"),
                color=alt.Color("metric:N", title="Metric"),
                tooltip=[
                    alt.Tooltip("threshold:Q", title="Threshold"),
                    alt.Tooltip("metric:N", title="Metric"),
                    alt.Tooltip("value:Q", title="Value"),
                ],
            )
            .properties(height=320),
            use_container_width=True,
        )

    st.subheader("Placement Risk Quadrant")
    risk = diagnostics["risk_quadrants"]
    if not risk.empty:
        st.dataframe(risk, use_container_width=True, hide_index=True)
    predictions = diagnostics["predictions"]
    if not predictions.empty:
        st.altair_chart(
            alt.Chart(predictions.sample(min(len(predictions), 2000), random_state=42))
            .mark_circle(size=45, opacity=0.35)
            .encode(
                x=alt.X("predicted_adoption_probability:Q", title="Predicted adoption probability"),
                y=alt.Y("predicted_days_to_outcome:Q", title="Predicted days to outcome"),
                color="animal_type:N",
                tooltip=["animal_type", "age_group", "intake_type", "predicted_adoption_probability", "predicted_days_to_outcome"],
            )
            .properties(height=360),
            use_container_width=True,
        )
    st.subheader("Error Slice Explorer")
    st.dataframe(diagnostics["classification_slices"].head(20), use_container_width=True, hide_index=True)
    st.dataframe(diagnostics["regression_slices"].head(20), use_container_width=True, hide_index=True)

with tabs[7]:
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

with tabs[8]:
    st.subheader("Campaign Candidate Finder")
    st.caption("Exploratory cohort finder for groups that may benefit from targeted visibility. This is not causal recommendation logic.")
    predictions = diagnostics["predictions"]
    if predictions.empty:
        st.info("Run diagnostics to populate campaign cohorts.")
    else:
        filters = {}
        cols = st.columns(4)
        for col, field in zip(cols, ["animal_type", "age_group", "intake_type", "covid_period"]):
            options = ["All"] + sorted(predictions[field].dropna().astype(str).unique().tolist())
            filters[field] = col.selectbox(field.replace("_", " ").title(), options)
        cohort = predictions.copy()
        for field, value in filters.items():
            if value != "All":
                cohort = cohort[cohort[field].astype(str).eq(value)]
        if cohort.empty:
            st.warning("No records match this cohort.")
        else:
            cols = st.columns(4)
            cols[0].metric("Cohort size", f"{len(cohort):,}")
            cols[1].metric("Observed adoption", f"{cohort['classification_target'].mean() * 100:.1f}%")
            cols[2].metric("Mean predicted adoption", f"{cohort['predicted_adoption_probability'].mean() * 100:.1f}%")
            cols[3].metric("Median predicted days", f"{cohort['predicted_days_to_outcome'].median():.1f}")
            st.write(
                "Campaign framing: this cohort may be useful for targeted visibility when predicted adoption probability is low "
                "or predicted days to outcome are high. Treat this as a prioritization signal, not proof of intervention impact."
            )
            st.dataframe(cohort.head(100), use_container_width=True, hide_index=True)

with tabs[9]:
    st.subheader("What-if Prediction")
    st.caption("Uses the combined CatBoost classifier and regressor when advanced artifacts exist. This is a demo prediction, not a causal decision rule.")

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
            similar = similar_historical_cases(DATA_PATH, record)
            if not similar.empty:
                st.subheader("Similar Historical Cases")
                st.dataframe(similar, use_container_width=True, hide_index=True)
        except FileNotFoundError as error:
            st.error(str(error))
            st.info("Run `python scripts/train_advanced.py --data data/processed/modeling_dataset.csv` first.")

with tabs[10]:
    st.subheader("Adoption Timeline")
    milestones = tables["milestones"]
    if milestones.empty:
        st.info("Run diagnostics to generate adoption timeline milestones.")
    else:
        group = st.selectbox("Timeline group", sorted(milestones["group"].unique()))
        view = milestones[milestones["group"].eq(group)].head(15)
        timeline_chart = view.melt(
            id_vars=["value", "adoptions"],
            value_vars=["adopted_by_day_7_pct", "adopted_by_day_30_pct", "adopted_by_day_90_pct"],
            var_name="milestone",
            value_name="share",
        )
        st.altair_chart(
            alt.Chart(timeline_chart)
            .mark_bar()
            .encode(
                x=alt.X("value:N", sort="-y", title=group.replace("_", " ")),
                y=alt.Y("share:Q", title="Share adopted (%)"),
                color=alt.Color("milestone:N", title="Milestone"),
                tooltip=[
                    alt.Tooltip("value:N", title="Value"),
                    alt.Tooltip("adoptions:Q", title="Adoptions"),
                    alt.Tooltip("milestone:N", title="Milestone"),
                    alt.Tooltip("share:Q", title="Adopted by day"),
                ],
            )
            .properties(height=360),
            use_container_width=True,
        )
        figure(FIGURES_DIR / "adoption_cumulative_curves.png", "Adoption timeline milestones")
        st.dataframe(milestones, use_container_width=True, hide_index=True)

with tabs[11]:
    st.subheader("Generated Artifacts")
    st.write("Core commands:")
    st.code(
        "\n".join(
            [
                "python scripts/build_dataset.py --intakes data/raw/intakes.csv --outcomes data/raw/outcomes.csv --output data/processed/modeling_dataset.csv",
                "python scripts/run_eda.py --data data/processed/modeling_dataset.csv",
                "python scripts/train_baseline.py --data data/processed/modeling_dataset.csv",
                "python scripts/train_boosting.py --data data/processed/modeling_dataset.csv",
                "python scripts/train_advanced.py --data data/processed/modeling_dataset.csv",
                "python scripts/run_analysis.py --data data/processed/modeling_dataset.csv",
                "python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv --include-shap",
                "python scripts/generate_animal_research.py --data data/processed/modeling_dataset.csv",
                "python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv",
                "python scripts/generate_report_outputs.py",
            ]
        ),
        language="bash",
    )
    st.write("Reports directory:", str(PROJECT_ROOT / "reports"))
    st.write("Models directory:", str(MODELS_DIR))
