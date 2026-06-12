# Results Reference — AAC Adoption ML Pipeline

Current reproducible results, hypothesis findings, interpretation notes, and reproduction commands. Updated as of 2026-06-06.

See [`METHODOLOGY.md`](METHODOLOGY.md) for target definitions and causal-language rules before citing any numbers.

---

## Dataset Snapshot

| Property | Value |
|----------|-------|
| Source | Austin Animal Center (AAC), 2013–2025 |
| Processed file | `data/processed/modeling_dataset.csv` |
| Matched episodes | **162,390** intake/outcome rows |
| Animal types | Dogs and cats only |
| Split strategy | Time-aware chronological |
| Train period | 2013–2021 |
| Validation period | 2022–2023 |
| Test period | 2024–2025 |
| Leakage control | Model features are intake-time-only |

Raw files:

```
data/raw/intakes.csv
data/raw/outcomes.csv
data/raw/context/austin_weather_daily.csv          (optional)
data/raw/context/austin_311_animal_requests.csv    (optional)
```

Raw data and generated outputs are gitignored.

---

## Classification Results

Primary metric: **ROC-AUC** (test split). Secondary: PR-AUC.

Comparison file: `reports/tables/model_comparison_classification.csv`

| Subset | Best model | ROC-AUC (test) |
|--------|-----------|----------------|
| Combined | `hist_gradient_boosting` | ≈ 0.840 |
| Dogs | `hist_gradient_boosting` | ≈ 0.806 |
| Cats | `hist_gradient_boosting` | ≈ 0.865 |

**Interpretation:**
- Gradient boosting currently performs best by ROC-AUC across all subsets.
- Logistic regression remains useful as an interpretable baseline.
- Cats appear slightly easier to classify than dogs in the current feature setup.
- PR-AUC is tracked alongside ROC-AUC; see `model_comparison_classification.csv` for full values.

---

## Regression Results

Primary metric: **MAE** on `regression_target_days` (length of stay until any outcome).  
**This is not adoption speed.** See [`METHODOLOGY.md`](METHODOLOGY.md#target-variable-definitions).

Comparison file: `reports/tables/model_comparison_regression.csv`

| Subset | Best model | MAE (test, days) |
|--------|-----------|-----------------|
| Combined | `catboost` | ≈ 18.55 |
| Dogs | `catboost` | ≈ 21.56 |
| Cats | `catboost` | ≈ 15.79 |

> Regression MAE label in all reports: *"combined regression MAE: X days for length-of-stay / days-to-outcome prediction"*

**Interpretation:**
- Regression is harder than classification.
- Dummy median remains a meaningful baseline because length of stay is right-skewed.
- MAE is emphasized over R² because it is easier to explain operationally.
- CatBoost outperforms histogram boosting on regression; histogram boosting outperforms on classification.

---

## Hypothesis Findings

### H1 — Intake Circumstances vs. Appearance (Central)

**Status:** Supported descriptively and predictively.

Key tables: `reports/tables/h1_feature_family_importance.csv`, `reports/tables/h1_feature_family_ablation.csv` (if run with `--h1-ablation`)  
Key figures: `reports/figures/h1_feature_family_importance.png`, `reports/figures/h1_intake_type_adoption_rate.png`, `reports/figures/h1_intake_condition_adoption_rate.png`  
Narrative: `reports/summary/h1_interpretation.md`

Current descriptive pattern:
- `Owner Surrender` and `Abandoned` records show relatively high adoption rates.
- `Public Assist` and `Euthanasia Request` show much lower adoption rates.
- Intake-related features have stronger model-importance signal than appearance features.

### H3 — Age and Adoption Timing (Central)

**Status:** Supported descriptively and predictively.

Key tables: `reports/tables/h3_age_evidence_matrix.csv`, `reports/tables/h3_adopted_only_age_speed.csv`  
Key figures: `reports/figures/h3_age_adoption_rate.png`, `reports/figures/h3_age_adopted_only_median_days.png`, `reports/figures/h3_age_shap_summary.png`  
Narrative: `reports/summary/h3_interpretation.md`

Current descriptive pattern:
- Baby animals have the highest adoption rate.
- Young animals are below babies but above adults.
- Adult and senior animals show lower adoption rates.
- Age-related features have meaningful SHAP importance.
- For adopted animals specifically, see `h3_adopted_only_age_speed.csv` for median `days_to_adoption`.

### H5 — COVID-Period Adoption Dynamics (Central)

**Status:** Supported descriptively.

Key tables: `reports/tables/h5_covid_evidence_matrix.csv`, `reports/tables/h5_covid_population_mix.csv`  
Key figures: `reports/figures/h5_covid_adoption_rate.png`, `reports/figures/h5_covid_median_los.png`, `reports/figures/h5_covid_intake_volume.png`  
Narrative: `reports/summary/h5_interpretation.md`

Current descriptive pattern:
- Post-COVID and COVID periods show higher adoption rates than pre-COVID.
- Median days to outcome are also higher in COVID and post-COVID periods.
- Caution: period effects may reflect policy changes, intake volume shifts, public behavior, and population mix — not a causal COVID effect.

### H2 — Seasonality (Secondary, descriptive)

**Status:** Descriptive check only.

Tables: `reports/tables/h2_seasonality_summary.csv`  
Figures: `reports/figures/h2_adoption_rate_by_season.png`, `reports/figures/h2_median_los_by_season.png`  
Narrative: `reports/summary/h2_interpretation.md`

### H4 — Dark Coat Color / Black Dog Syndrome (Secondary, descriptive)

**Status:** Descriptive check only.

Tables: `reports/tables/h4_dark_color_summary.csv`  
Figures: `reports/figures/h4_dark_color_adoption_rate.png`, `reports/figures/h4_dark_color_median_los.png`  
Narrative: `reports/summary/h4_interpretation.md`

Note: `is_black_or_dark` is an approximate operational grouping. Dark-colored animals show lower observed adoption rates — this is a descriptive association consistent with shelter literature, not proof of discrimination.

---

## External Context Feature Test

The context workflow compares `intake_time_v1` (base) against `intake_time_context_v1` (enriched with weather, 311, and shelter-volume features).

Comparison artifact:
```
reports/tables/context_model_comparison.csv
reports/figures/context_model_delta.png
```

**Current interpretation:**
- Context features gave small CatBoost classification gains for combined animals and cats, but not dogs.
- Context features gave small CatBoost regression MAE gains across combined, dog, and cat subsets.
- Context features did not broadly improve every model family; several random-forest and histogram-boosting regression runs worsened.
- Thesis wording: *"External context added limited but useful signal for CatBoost; it did not generally improve all model families."*

---

## Interpretability Outputs

Generated by `scripts/generate_diagnostics.py --include-shap` and `scripts/generate_feature_family_importance.py`.

| File | Content |
|------|---------|
| `reports/tables/logistic_regression_coefficients.csv` | LR coefficient table |
| `reports/tables/random_forest_feature_importance.csv` | RF importance |
| `reports/tables/permutation_importance_classification.csv` | Permutation importance (classification) |
| `reports/tables/permutation_importance_regression.csv` | Permutation importance (regression) |
| `reports/tables/shap_global_classification.csv` | SHAP global feature values (classification) |
| `reports/tables/shap_global_regression.csv` | SHAP global feature values (regression) |
| `reports/tables/shap_feature_families_classification.csv` | SHAP by feature family (classification) |
| `reports/tables/shap_feature_families_regression.csv` | SHAP by feature family (regression) |

**Recommended thesis language:**
> *The model identifies variables associated with adoption outcomes and length of stay. These associations do not prove causal effects, but they indicate which intake-time characteristics are most informative for prediction. SHAP values explain how features contributed to this model's prediction. They do not prove that changing a feature would causally change adoption probability.*

---

## Model Reliability and Evidence Pack Outputs

Generated by `scripts/generate_evidence_pack.py`.

| File | Content |
|------|---------|
| `reports/tables/model_evidence_pack.csv` | Summary of model choice, PR-AUC, bootstrap intervals |
| `reports/tables/metric_confidence_intervals.csv` | Bootstrap metric CIs |
| `reports/tables/model_limitations_by_cohort.csv` | Cohort-level error and calibration limits |
| `reports/tables/subgroup_reliability.csv` | Per-cohort reliability metrics |
| `reports/tables/subgroup_metric_confidence_intervals.csv` | Per-cohort bootstrap CIs |
| `reports/tables/subgroup_adoption_milestones.csv` | Descriptive adoption timing at 7/30/60/90 days |
| `reports/tables/model_failure_modes.csv` | Ranked calibration gaps, MAE, FNR, FPR by cohort |
| `reports/tables/animal_journey_examples.csv` | Representative Animal Journey Card examples |
| `reports/tables/local_explanation_examples.csv` | Local explanation examples (non-causal) |
| `reports/summary/model_evidence_pack.md` | Human-readable evidence pack narrative |
| `reports/summary/subgroup_reliability.md` | Subgroup reliability narrative |

**Top subgroup reliability fields:**
- `records` — cohort size
- `observed_adoption_rate` — actual share adopted
- `mean_predicted_probability` — average model prediction
- `calibration_gap` — absolute difference between observed and predicted rates
- `small_cohort` — flag for groups too small for strong claims

`model_failure_modes.csv` ranks the largest calibration gaps, MAE values, false-negative rates, and false-positive rates. High-gap rows are warning lights for thesis discussion, not proof that a group is intrinsically harder to place.

---

## Interpretation Rules

- Use **ROC-AUC** for ranking classification quality.
- Use **PR-AUC** for adoption-positive precision/recall behavior.
- Use **calibration and threshold tables** before interpreting probabilities as decision-support scores.
- Use **MAE** as the main days-to-outcome regression metric.
- Use **SHAP** as model-explanation evidence only: features are associated with model predictions, not causes of adoption.
- Treat health and behavior fields as administrative care-context proxies, not complete personality or temperament labels.
- Treat animal profile findings as descriptive/exploratory unless supported by additional statistical testing.
- **Do not interpret subgroup metrics below minimum sample thresholds** (n < 100 or n < 200 depending on context).

---

## Full Reproduction Commands

```bash
python scripts/download_raw_data.py --source historical --output-dir data/raw --overwrite
python scripts/download_context_data.py --output-dir data/raw/context
python scripts/build_dataset.py --intakes data/raw/intakes.csv --outcomes data/raw/outcomes.csv --output data/processed/modeling_dataset.csv
python scripts/build_dataset.py --intakes data/raw/intakes.csv --outcomes data/raw/outcomes.csv --output data/processed/modeling_dataset_context.csv --context-data-dir data/raw/context
python scripts/run_eda.py --data data/processed/modeling_dataset.csv
python scripts/train_baseline.py --data data/processed/modeling_dataset.csv
python scripts/train_boosting.py --data data/processed/modeling_dataset.csv
python scripts/train_advanced.py --data data/processed/modeling_dataset.csv
python scripts/train_baseline.py --data data/processed/modeling_dataset_context.csv --metrics-dir reports/metrics_context --models-dir models/context_baseline --tables-dir reports/tables_context --output reports/metrics_context/baseline_metrics.csv
python scripts/train_boosting.py --data data/processed/modeling_dataset_context.csv --metrics-dir reports/metrics_context --models-dir models/context_boosting --tables-dir reports/tables_context
python scripts/train_advanced.py --data data/processed/modeling_dataset_context.csv --metrics-dir reports/metrics_context --models-dir models/context_advanced
python scripts/run_analysis.py --data data/processed/modeling_dataset.csv
python scripts/compare_context_models.py --base-metrics-dir reports/metrics --context-metrics-dir reports/metrics_context --tables-dir reports/tables
python scripts/generate_diagnostics.py --data data/processed/modeling_dataset.csv --include-shap
python scripts/generate_animal_research.py --data data/processed/modeling_dataset.csv
python scripts/generate_evidence_pack.py --data data/processed/modeling_dataset.csv
python scripts/generate_report_outputs.py
pytest
```

---

## Current Test Status

```bash
pytest -q
```

Latest verified result: **209 passed** (local environment, 2026-06-12).

Known warning: `FutureWarning` in `evidence_pack.py` during pandas concat with empty/all-NA frames in synthetic tests. Does not fail the suite.
