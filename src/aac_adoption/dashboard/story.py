"""Narrative dashboard helpers for approach comparison and workflow visuals."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def approach_comparison_rows() -> pd.DataFrame:
    """Return thesis-friendly comparison of analytical layers."""
    return pd.DataFrame(
        [
            {
                "layer": "Baseline models",
                "technology": "Dummy, Logistic Regression, Ridge",
                "answers": "What is minimum credible performance?",
                "strength": "Simple, explainable reference point",
                "dashboard_use": "Model comparison baseline",
            },
            {
                "layer": "Tree ensembles",
                "technology": "Random Forest, Histogram Gradient Boosting",
                "answers": "How much nonlinear signal exists?",
                "strength": "Strong tabular performance",
                "dashboard_use": "Main ROC-AUC / MAE comparison",
            },
            {
                "layer": "Advanced tabular ML",
                "technology": "CatBoost",
                "answers": "Can native categorical modeling improve outcomes?",
                "strength": "Handles shelter categories directly",
                "dashboard_use": "Advanced model and sensitivity demo",
            },
            {
                "layer": "Reliability diagnostics",
                "technology": "Calibration, thresholds, residuals",
                "answers": "Can users trust probabilities and errors?",
                "strength": "Shows confidence and failure modes",
                "dashboard_use": "Model Quality and Risk Explorer",
            },
            {
                "layer": "Interpretability",
                "technology": "SHAP, feature families",
                "answers": "What drives predictions?",
                "strength": "Explains predictive associations",
                "dashboard_use": "Interpretability tab",
            },
            {
                "layer": "Decision storytelling",
                "technology": "Risk quadrants, similar cases, campaigns",
                "answers": "What could shelter staff do with this?",
                "strength": "Turns scores into operational signals",
                "dashboard_use": "Campaign Finder and Sensitivity Demo",
            },
        ]
    )


def workflow_dot() -> str:
    """Return Graphviz workflow for Streamlit rendering."""
    return """
digraph {
  graph [rankdir=LR, bgcolor="transparent"];
  node [shape=box, style="rounded,filled", color="#d6dee8", fillcolor="#f8fafc", fontname="Arial"];
  edge [color="#64748b", arrowsize=0.8];

  raw [label="Raw AAC\\nintakes + outcomes"];
  clean [label="Clean + match\\nshelter episodes"];
  features [label="Leakage-safe\\nintake features"];
  models [label="Baselines +\\nBoosting + CatBoost"];
  diagnostics [label="Calibration\\nerrors + SHAP"];
  story [label="Dashboard\\nstory views"];
  decisions [label="Thesis evidence +\\nshelter signals"];

  raw -> clean -> features -> models -> diagnostics -> story -> decisions;
}
"""


def decision_sankey() -> go.Figure:
    """Return data-to-decision Sankey figure."""
    labels = [
        "AAC records",
        "Episode dataset",
        "Adoption model",
        "LOS model",
        "Reliability checks",
        "SHAP explanations",
        "Risk quadrants",
        "Campaign candidates",
        "Model sensitivity demo",
    ]
    source = [0, 1, 1, 2, 3, 2, 4, 5, 6]
    target = [1, 2, 3, 4, 4, 5, 6, 7, 8]
    value = [10, 5, 5, 3, 3, 2, 3, 2, 2]
    fig = go.Figure(
        data=[
            go.Sankey(
                node={"label": labels, "pad": 18, "thickness": 16},
                link={"source": source, "target": target, "value": value},
            )
        ]
    )
    fig.update_layout(height=360, margin={"l": 10, "r": 10, "t": 10, "b": 10})
    return fig


def story_cards() -> list[dict[str, str]]:
    """Return real-life story cards for dashboard display."""
    return [
        {
            "title": "Probability Trust",
            "question": "When model says 70% adoption chance, does reality agree?",
            "artifact": "Calibration curve",
        },
        {
            "title": "Long-stay Risk",
            "question": "Which animals look adoptable but may wait longer?",
            "artifact": "Placement Risk Quadrant",
        },
        {
            "title": "Model Failure Modes",
            "question": "Where do false negatives and large LOS errors cluster?",
            "artifact": "Error Slice Explorer",
        },
        {
            "title": "Campaign Finder",
            "question": "Which cohorts may deserve targeted visibility?",
            "artifact": "Campaign Candidate Finder",
        },
        {
            "title": "Similar Cases",
            "question": "What happened historically to animals like this one?",
            "artifact": "Similar Historical Cases",
        },
    ]
