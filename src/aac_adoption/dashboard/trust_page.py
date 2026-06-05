"""Trust & Limits page content and layout for the streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

def render_trust_and_limits(tables: dict[str, pd.DataFrame]) -> None:
    """Render the Trust & Limits page in the Streamlit app.
    
    Provides critical methodology context, causal disclaimers,
    and subgroup performance reliability disclosures.
    """
    st.subheader("🔍 Trust & Limits — Methodological Disclosures")
    
    # Large Causal Warning at the top
    st.warning(
        "⚠️ **Methodological disclaimer:**\n\n"
        "SHAP values explain how features contributed to this model's prediction. "
        "They do not prove that changing a feature would causally change adoption probability. "
        "Do not interpret predictive associations as real-world causes."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Scope: What the Model Predicts")
        st.markdown(
            "1. **Binary Adoption Outcome:** Predicts whether an animal will be adopted "
            "(vs. transfer, return to owner, euthanasia, etc.) using shelter intake data.\n"
            "2. **Predicted Length of Stay (LOS):** For animals that are adopted, predicts the "
            "number of days they will spend in the shelter until the adoption event."
        )
        
        st.markdown("### ⛔ What the Model Does NOT Predict")
        st.markdown(
            "- **Individual Animal Fate:** The model provides statistical probabilities and "
            "expected timelines based on historical averages, not individual guarantees.\n"
            "- **Causal Intervention Impact:** Changing a feature in the Model Sensitivity Demo (e.g. changing "
            "an animal's name or reproductive status) does not prove that the real-world outcome "
            "would change accordingly.\n"
            "- **Out-of-Jurisdiction Generalisation:** The model is trained exclusively on "
            "Austin Animal Center (AAC) records and may not generalise to other geographical areas."
        )
        
    with col2:
        st.markdown("### ⏳ Intake-Time-Only Constraint")
        st.markdown(
            "This model operates exclusively on features available **at the time of intake** "
            "(e.g. intake age, sex, breed, intake condition, and weather/load context).\n\n"
            "**Why this matters:** It cannot account for events that occur *after* the animal "
            "enters the shelter (e.g. behavioral issues discovered later, medical treatments, "
            "or promotional campaigns). This design prevents data leakage and ensures the model "
            "can be used immediately when an animal is processed."
        )
        
        st.markdown("### 🌍 External Validity warning")
        st.markdown(
            "All training data is sourced from the **Austin Animal Center (AAC)** in Austin, Texas.\n\n"
            "Shelter policies, local stray dynamics, breed popularity, and resource constraints "
            "vary widely by region. Users should exercise extreme caution if applying these "
            "insights to shelter systems outside of Austin."
        )

    st.markdown("---")
    
    # Subgroup Reliability Disclosures
    st.markdown("### ⚠️ Cohort Reliability & Performance Red Flags")
    st.markdown(
        "Standard machine learning metrics (like aggregate ROC-AUC or MAE) can hide severe "
        "performance drops in smaller or more vulnerable animal cohorts. Below are identified reliability red flags."
    )
    
    subgroup_df = tables.get("subgroup_reliability")
    if subgroup_df is not None and not subgroup_df.empty:
        # Filter out small cohorts or show them with a warning
        flagged = subgroup_df.copy()
        if "calibration_gap" in flagged.columns:
            # Highlight cohorts with large calibration gap
            high_gap = flagged[flagged["calibration_gap"] > 0.08]
            if not high_gap.empty:
                st.write("**High Calibration Discrepancy (Gap > 8%):**")
                st.dataframe(high_gap[["cohort", "value", "records", "observed_adoption_rate", "calibration_gap"]], use_container_width=True, hide_index=True)
            else:
                st.info("No cohorts exhibit a calibration gap greater than 8%.")
    else:
        st.info("Run `python scripts/generate_evidence_pack.py` to populate cohort reliability data.")
