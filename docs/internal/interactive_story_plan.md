# Interactive Story and Approach Comparison Plan

This plan adds a thesis-defense story layer on top of the advanced ML and diagnostics system.

## Goal

Make the app explain the project as a complete analytical workflow:

1. raw AAC data becomes leakage-safe model data,
2. baseline and advanced models compete under the same time split,
3. diagnostics show trust and failure modes,
4. SHAP explains model behavior,
5. shelter-facing views translate model outputs into practical decision signals.

## Story Structure

Add a Streamlit `Story Mode` tab before technical tabs.

Story sections:

- `Pipeline Map`: workflow from raw data to dashboard.
- `Approach Comparison`: baseline vs boosting vs CatBoost vs diagnostics vs SHAP.
- `Why This Matters`: connect each technical layer to real shelter questions.
- `Decision Journey`: example path from animal intake to risk quadrant, similar cases, and campaign candidate view.

## Visualization Ideas

Use multiple visual grammars:

- Graph workflow for data and model lifecycle.
- Sankey-style flow for data-to-decision story.
- Altair comparison chart for model approach strengths.
- Compact cards for real-life use cases:
  - probability trust,
  - long-stay risk,
  - similar historical cases,
  - campaign candidate finder.

## Implementation

Create `src/aac_adoption/dashboard/story.py`.

Expose:

- `approach_comparison_rows()`
- `workflow_dot()`
- `decision_sankey()`
- `story_cards()`

Update `streamlit_app.py`:

- add `Story Mode` tab,
- render workflow chart,
- render approach comparison,
- render decision-support story cards,
- keep all content artifact-driven where possible.

## Acceptance

- `python -m pytest` passes.
- Streamlit app loads.
- `Story Mode` visible.
- Story explains why CatBoost, diagnostics, SHAP, thresholding, risk quadrants, and similar cases exist.
- No model retraining happens inside dashboard.
