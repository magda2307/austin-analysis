# Priority 3 & 4 — Academic Defence & Model Evaluation

Strengthen the thesis with a hypothesis evidence matrix, per-hypothesis deep dives, and a clean model-selection narrative. All outputs are pure Python/pandas/matplotlib — no model retraining is needed except for the optional H1 ablation study.

---

## Open Questions

> [!IMPORTANT]
> **H1 Ablation Study (Task 3.2 — strongest version)**
> Training six separate feature-family models requires re-running `train_all_baselines` / `train_all_boosting` six times. Each full run takes ~5–10 min on your machine. Should the agent run the ablation automatically, or skip it and produce only the permutation-importance + SHAP family tables (which already exist)?
>
> Options:
> - **A — Run ablation automatically** (produces `h1_feature_family_ablation.csv` from actual models; adds ~30–60 min total).
> - **B — Skip ablation, rely on existing SHAP/permutation importance** (faster; ablation table will be a clear placeholder or omitted).

> [!NOTE]
> **PR-AUC (Task 4.2)**  
> `pr_auc` is already computed in `evaluate.py` and in the existing metrics CSVs (the column exists but some files might have `None` for baselines that lacked scores). The agent will backfill from existing CSVs and add the comparison figure — no model retraining needed.

---

## Proposed Changes

### Priority 3 — Hypothesis Evidence

---

#### Task 3.1 — Hypothesis Evidence Matrix

**New module + entry-point script. Pure analysis from existing artifacts.**

##### [NEW] `src/aac_adoption/analysis/hypothesis_evidence.py`
Central function `create_hypothesis_evidence_matrix()` that:
- Reads existing tables (`shap_feature_families_*.csv`, `permutation_importance_*.csv`, `h1_intake_vs_appearance.csv`, `h3_age_adoption_speed.csv`, `h5_covid_period.csv`, `model_evidence_pack.csv`, `subgroup_reliability.csv`)
- Assembles one row per hypothesis (H1–H5) with columns:
  - `hypothesis`, `status`, `primary_evidence`, `descriptive_evidence_file`, `model_evidence_file`, `interpretability_evidence_file`, `reliability_caveat`, `final_interpretation`, `causal_warning`
- Writes `reports/tables/hypothesis_evidence_matrix.csv`
- Writes `reports/summary/hypothesis_evidence_matrix.md` (narrative table + per-hypothesis paragraph)

Status vocabulary: `supported_descriptively`, `supported_predictively`, `partially_supported`, `not_supported`, `inconclusive`.

---

#### Task 3.2 — H1: Intake Circumstances vs. Appearance

**New module + figures. Uses existing importance data; optionally runs ablation.**

##### [NEW] `src/aac_adoption/analysis/h1_feature_family.py`
- `create_h1_feature_family_importance()` — aggregates permutation importance + SHAP means by the five families defined in the spec:
  - `intake_context`, `appearance`, `age`, `calendar`, `identity`
  - Outputs `reports/tables/h1_feature_family_importance.csv`
  - Outputs `reports/figures/h1_feature_family_importance.png` (horizontal bar chart, dogs vs cats side-by-side)
- `create_h1_ablation_table()` (optional / gated by flag) — trains six `HistGradientBoostingClassifier` variants with restricted feature sets, records test ROC-AUC + F1 per subset (dogs/cats/combined), writes `reports/tables/h1_feature_family_ablation.csv`
- `create_h1_interpretation_md()` — writes `reports/summary/h1_interpretation.md`

##### [MODIFY] `scripts/run_analysis.py`
Add `--h1-ablation` flag that triggers the ablation path.

---

#### Task 3.3 — H3: Age and Adopted-Only Timing

**New module + three-level evidence figures.**

##### [NEW] `src/aac_adoption/analysis/h3_age_evidence.py`
- **Level 1** — adoption probability by age group (dogs / cats separately). Reads `modeling_dataset.csv`. Writes `reports/tables/h3_age_evidence_matrix.csv` and `reports/figures/h3_age_adoption_rate.png`.
- **Level 2** — adopted-only median days (filter `adopted == 1`; uses `days_to_adoption` if present, else `days_to_outcome`). Writes `reports/figures/h3_age_adopted_only_median_days.png`.
- **Level 3** — SHAP age-feature summary pulled from `shap_global_classification.csv` / `shap_global_regression.csv`. Writes `reports/figures/h3_age_shap_summary.png`.
- `create_h3_interpretation_md()` — writes `reports/summary/h3_interpretation.md`.

---

#### Task 3.4 — H5: COVID-Period Change

**New module + six mini-tables + four figures.**

##### [NEW] `src/aac_adoption/analysis/h5_covid_evidence.py`
- Reads `modeling_dataset.csv`; groups by `covid_period` (`pre_covid`, `covid`, `post_covid`)
- Produces:
  - `reports/tables/h5_covid_evidence_matrix.csv` — adoption rate / median LOS / intake volume by period
  - `reports/tables/h5_covid_population_mix.csv` — species mix, intake type mix, age group mix by period
  - `reports/figures/h5_covid_adoption_rate.png`
  - `reports/figures/h5_covid_median_los.png`
  - `reports/figures/h5_covid_intake_volume.png`
  - `reports/summary/h5_interpretation.md` — explicitly says "associated with changed patterns", not "caused"

---

#### Task 3.5 — H2 and H4 Secondary Outputs

**Extend existing `hypothesis_tables.py` with two new functions.**

##### [MODIFY] `src/aac_adoption/analysis/hypothesis_tables.py`
Add:
- `create_h2_seasonality_outputs()` — groups by `intake_season`; outputs:
  - `reports/tables/h2_seasonality_summary.csv`
  - `reports/figures/h2_adoption_rate_by_season.png`
  - `reports/figures/h2_median_los_by_season.png`
- `create_h4_dark_color_outputs()` — groups by `is_black_or_dark`; outputs:
  - `reports/tables/h4_dark_color_summary.csv`
  - `reports/figures/h4_dark_color_adoption_rate.png`
  - `reports/figures/h4_dark_color_median_los.png`
  - Both figures include a subtitle note: *"is_black_or_dark is an approximate operational grouping"*

---

### Priority 4 — Model Evaluation

---

#### Task 4.1 — Final Model Selection Report

**New analysis module. Pure read + decision logic from existing CSVs.**

##### [NEW] `src/aac_adoption/analysis/model_selection.py`
- `create_final_model_selection()` reads `model_comparison_classification.csv` and `model_comparison_regression.csv`
- Applies selection rule: classification — highest ROC-AUC among test rows, tie-break by PR-AUC; note calibration/interpretability justification; regression — lowest MAE
- Marks winner `selected=True`; adds `selection_reason` text column
- Outputs `reports/tables/final_model_selection.csv` and `reports/summary/final_model_selection.md`

---

#### Task 4.2 — PR-AUC Everywhere

**Backfill + new comparison figure. No retraining.**

##### [MODIFY] `src/aac_adoption/analysis/model_comparison.py`
- Add `pr_auc` column to comparison ranking (already exists in source CSVs for boosting/advanced models; baseline CSVs may have `None` for dummy)
- Add `pr_auc_rank` alongside existing `roc_auc_rank`
- Re-save `reports/tables/model_comparison_classification.csv`

##### [NEW] figure in `create_model_comparison_tables()`
- `reports/figures/model_comparison_classification_pr_auc.png` — mirroring existing `roc_auc` figure

##### [MODIFY] `reports/summary/current_results.md`
- Add PR-AUC column to the model comparison table section

---

#### Task 4.3 — Confusion Matrix and Threshold Discussion

**New analysis reading saved model artifact for the selected classifier.**

##### [NEW] `src/aac_adoption/analysis/threshold_analysis.py`
- Loads best classifier (hist_gradient_boosting or catboost — whichever `final_model_selection.csv` marks selected) via `joblib`
- Computes predicted probabilities on test set
- Evaluates four thresholds: 0.50, F1-max, recall-focused (≥0.85 recall), balanced
- Outputs:
  - `reports/tables/final_classifier_thresholds.csv` — per-threshold: precision, recall, F1, FPR, FNR
  - `reports/figures/final_confusion_matrix.png` — 2×2 confusion matrix at default + F1-max thresholds (side-by-side subplots)
  - `reports/summary/threshold_selection.md` — explicitly states threshold choice depends on operational goal

> [!NOTE]
> This requires loading the saved `.joblib` model file and the test split. The `make_time_split` function and the dataset CSV are available; this is a read-only run (no retraining).

---

#### Task 4.4 — Calibration Interpretation Summary

**New analysis reading the existing `diagnostic_calibration_curve` data.**

##### [NEW] `src/aac_adoption/analysis/calibration_summary.py`
- Reads `reports/metrics/*classification_metrics.csv` to get model-level calibration gap (from `subgroup_reliability.csv` and existing calibration diagnostic)
- Reads per-cohort calibration from `subgroup_reliability.csv`
- Answers the five questions from the spec (overconfident? underconfident? reliable bins? dogs vs cats? probabilities as ranking vs literal truth?)
- Outputs:
  - `reports/tables/calibration_summary_by_subset.csv` — per animal_type + model: mean calibration gap, worst cohort
  - `reports/summary/calibration_interpretation.md`

---

#### Task 4.5 — Subgroup Reliability Red Flags

**New analysis reading `subgroup_reliability.csv` (already populated).**

##### [NEW] `src/aac_adoption/analysis/reliability_red_flags.py`
- Reads `subgroup_reliability.csv`
- Derives `false_positive_rate`, `false_negative_rate`, `regression_mae`, `small_cohort_flag` (already present as columns)
- Adds `risk_level`: `high` (calibration_gap > 0.15 or small_cohort), `medium` (0.08–0.15), `low` (<0.08)
- Adds `interpretation` text column (e.g., "Model overestimates adoption likelihood for terrier-type dogs (gap = 0.21). Small cohort flag not set but gap is large.")
- Outputs:
  - `reports/tables/model_reliability_red_flags.csv`
  - `reports/summary/model_reliability_red_flags.md`

---

### Orchestration

##### [MODIFY] `scripts/run_analysis.py`
Wire all new functions into the main pipeline with CLI flags:
- `--h1-ablation` — enables ablation training (default: off)
- `--skip-model-load` — skips threshold analysis that requires loading joblib (default: on, since model artifacts exist)

All new functions added to the default run (except ablation).

---

## New File Summary

| File | Type | Task |
|------|------|------|
| `src/aac_adoption/analysis/hypothesis_evidence.py` | NEW | 3.1 |
| `src/aac_adoption/analysis/h1_feature_family.py` | NEW | 3.2 |
| `src/aac_adoption/analysis/h3_age_evidence.py` | NEW | 3.3 |
| `src/aac_adoption/analysis/h5_covid_evidence.py` | NEW | 3.4 |
| `src/aac_adoption/analysis/hypothesis_tables.py` | MODIFY | 3.5 |
| `src/aac_adoption/analysis/model_selection.py` | NEW | 4.1 |
| `src/aac_adoption/analysis/model_comparison.py` | MODIFY | 4.2 |
| `src/aac_adoption/analysis/threshold_analysis.py` | NEW | 4.3 |
| `src/aac_adoption/analysis/calibration_summary.py` | NEW | 4.4 |
| `src/aac_adoption/analysis/reliability_red_flags.py` | NEW | 4.5 |
| `scripts/run_analysis.py` | MODIFY | orchestration |

---

## Verification Plan

### Automated check
```
python scripts/run_analysis.py --data data/processed/modeling_dataset.csv
```
All outputs listed above must exist and be non-empty after the run.

### Manual verification
- `hypothesis_evidence_matrix.md` — each hypothesis has a row; no hypothesis is listed as `not_evaluated`
- `h1_feature_family_importance.png` — shows ≥3 families with bars for dogs and cats
- `h3_age_evidence_matrix.csv` — has one row per age group × animal_type
- `h5_covid_population_mix.csv` — has species mix + intake type + age group columns
- `final_model_selection.md` — clearly names the winning model with `selection_reason`
- `final_classifier_thresholds.csv` — has four rows (one per threshold)
- `model_reliability_red_flags.csv` — `risk_level` populated; small-cohort rows flagged
- PR-AUC column appears in `model_comparison_classification.csv`
