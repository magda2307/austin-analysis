# Architecture Reference — AAC Adoption ML Pipeline

This document describes the system design, pipeline layers, data and schema contracts, modeling decisions, technology stack, and test strategy. It is the technical reference for developers and AI agents extending this repository.

---

## System Design Philosophy

The repository follows an **artifact-first, leakage-safe, thesis-reproducible** architecture:

1. **Artifact-first:** every important analysis output is written to `reports/` as stable CSV, Markdown, PNG, or model files. The Streamlit app reads artifacts — it never retrains models or rebuilds datasets.
2. **Leakage-safe:** configured model features come only from intake-time data and prior context windows. `validate_no_leakage()` enforces this at build time.
3. **Date-aware:** intakes are matched to future outcomes only; the split is chronological. Random splits are never used for thesis evaluation.
4. **Model-comparison oriented:** five model families (dummy → linear → random forest → histogram boosting → CatBoost) are kept for methodological progression.
5. **Thesis-ready language:** all generated evidence uses predictive/descriptive wording. Causal claims are forbidden. See [`METHODOLOGY.md`](METHODOLOGY.md).

---

## Repository Layer Map

```
src/aac_adoption/
  data/           data loading, cleaning, matching, context features
  features/       leakage-safe feature engineering and feature registry
  models/         baseline, boosting, CatBoost training, splitting, metrics
  analysis/       hypothesis tables, H1/H3/H5 evidence, model selection, thresholds, calibration, reliability red flags
  diagnostics/    model diagnostics, SHAP summaries, feature-family diagnostics
  interpretation/ feature importance and explanation helpers
  reporting/      evidence-pack and thesis summary generation
  dashboard/      Streamlit data helpers, story visuals, trust/limits page

scripts/          command-line entrypoints (one per pipeline step) + orchestration runners
tests/            automated regression tests
docs/             METHODOLOGY.md, ARCHITECTURE.md, RESULTS.md, ROADMAP.md
data/raw/         raw AAC CSV files (gitignored)
data/processed/   modeling datasets and metadata (gitignored)
models/           trained model artifacts (gitignored, .gitkeep preserved)
reports/          generated tables, figures, summaries, diagnostics (gitignored, .gitkeep preserved)
```

---

## Pipeline Layers in Detail

### Layer 1: Data Acquisition

**Script:** `scripts/download_raw_data.py`, `scripts/download_context_data.py`

- Downloads raw AAC intake and outcome CSV exports from Austin Open Data (Socrata).
- Raw files are **never modified** by any downstream step.
- Optional context data: Austin daily weather and 311 animal-service demand.

**Sources:**
- Historical (2013-10-01 to 2025-05-05): `--source historical`
- Current (ShelterBuddy, 2025-05-05+): `--source current`

### Layer 2: Data Preparation

**Module:** `src/aac_adoption/data/`  
**Script:** `scripts/build_dataset.py`

Steps:
1. **Column normalization** — raw AAC column names converted to `snake_case` in `load_data.py`. `DateTime` becomes `intake_datetime` / `outcome_datetime`. Raw files are not modified.
2. **Cleaning** — required columns validated, duplicates removed, restricted to dogs and cats, mixed datetime formats parsed without shifting local shelter clock.
3. **Episode matching** — each intake matched to nearest unused future outcome for same `animal_id` (greedy nearest-future-match). Negative `days_to_outcome` rejected. Unmatched intakes counted.
4. **Target creation** — classification target (`classification_target`), regression target (`regression_target_days`), adoption-only timing (`days_to_adoption`).

**Required intake columns:** `animal_id`, `animal_type`, `intake_datetime`  
**Required outcome columns:** `animal_id`, `animal_type`, `outcome_datetime`, `outcome_type`

### Layer 3: Feature Engineering

**Module:** `src/aac_adoption/features/`

- Converts raw shelter variables into model-ready features.
- All features are intake-time-only (nothing from the outcome record).
- Feature registry: `src/aac_adoption/features/feature_sets.py` — contains `BASE_INTAKE_TIME_FEATURES`, `CATEGORICAL_FEATURES`, `LEAKAGE_COLUMNS`, and `validate_no_leakage()`.
- Feature metadata saved to `data/processed/feature_columns.json`.
- Target metadata saved to `data/processed/target_columns.json`.

**Found Location taxonomy** (implemented, geocoder-free):

The raw `Found Location` string is preserved in raw data but is not used directly as a model feature. Derived fields are deterministic and reproducible:

| Derived field | Values |
|--------------|--------|
| `found_location_kind` | `austin_city`, `county_or_region`, `outside_jurisdiction`, `intersection`, `address_like`, `other` |
| `found_location_area` | extracted city/county token |
| `is_austin_found_location` | boolean flag |
| `is_outside_jurisdiction` | boolean flag |
| `is_intersection_location` | boolean flag |
| `is_address_like_location` | boolean flag |
| `is_airport_location` | boolean flag |

Raw `found_location` is excluded from `BASE_INTAKE_TIME_FEATURES` to avoid high-cardinality raw location text in models.

### Layer 4: Modeling

**Module:** `src/aac_adoption/models/`  
**Scripts:** `train_baseline.py`, `train_boosting.py`, `train_advanced.py`

**Model families:**

| Family | Models | Script |
|--------|--------|--------|
| Dummy baselines | `DummyClassifier`, `DummyRegressor` | `train_baseline.py` |
| Linear baselines | `LogisticRegression`, `Ridge` | `train_baseline.py` |
| Tree baselines | `RandomForestClassifier`, `RandomForestRegressor` | `train_baseline.py` |
| Histogram boosting | `HistGradientBoostingClassifier/Regressor` | `train_boosting.py` |
| CatBoost | `CatBoostClassifier`, `CatBoostRegressor` | `train_advanced.py` |

**Animal subsets evaluated for each model:** `combined`, `dogs`, `cats`

**Default time-aware split:**

```
train:      2013–2021
validation: 2022–2023  (threshold selection, calibration fitting)
test:       2024–2025  (final untouched evaluation)
```

Falls back to deterministic random split if these years are unavailable.

**Saved artifacts:**

```
models/baseline/    baseline model pipelines (.joblib)
models/boosting/    histogram gradient boosting artifacts (.joblib)
models/advanced/    CatBoost artifacts (.cbm + metadata JSON)
```

Each artifact bundle includes feature lists and categorical feature lists.

### Layer 5: Analysis and Evidence

**Module:** `src/aac_adoption/analysis/`  
**Script:** `scripts/run_analysis.py`

Sub-modules and their outputs:

| Module | Key outputs |
|--------|------------|
| `model_comparison.py` | `model_comparison_classification.csv`, `model_comparison_regression.csv`, PR-AUC comparison figures |
| `model_selection.py` | `final_model_selection.csv`, `final_model_selection.md` |
| `threshold_analysis.py` | `final_classifier_thresholds.csv` (validation-selected, test-applied), `final_confusion_matrix.png` |
| `calibration_summary.py` | `calibration_summary_by_subset.csv`, `calibration_interpretation.md` |
| `reliability_red_flags.py` | `model_reliability_red_flags.csv`, `model_reliability_red_flags.md` |
| `hypothesis_evidence.py` | `hypothesis_evidence_matrix.csv`, `hypothesis_evidence_matrix.md` |
| `h1_feature_family.py` | `h1_feature_family_importance.csv`, `h1_interpretation.md`, optional ablation |
| `h3_age_evidence.py` | `h3_age_evidence_matrix.csv`, `h3_adopted_only_age_speed.csv`, `h3_interpretation.md` |
| `h5_covid_evidence.py` | `h5_covid_evidence_matrix.csv`, `h5_covid_population_mix.csv`, `h5_interpretation.md` |
| `hypothesis_tables.py` | H2 seasonality, H4 dark-color descriptive tables and figures |

**Diagnostics (selected-model-driven):**

`scripts/generate_diagnostics.py` resolves the selected model from `final_model_selection.csv` before producing calibration curves, error slices, and risk quadrants. It does not hardcode a model family.

**Evidence pack:**

`scripts/generate_evidence_pack.py` produces the ML-rigor layer: bootstrap metric intervals, cohort reliability, SHAP feature-family summaries, Animal Journey examples, local explanation examples, subgroup reliability, model failure modes, and adoption milestone tables.

### Layer 6: Presentation

**Script:** `streamlit_app.py`  
**Module:** `src/aac_adoption/dashboard/`

The app is a **read-only artifact consumer**. It does not retrain models.

Dashboard pages:
- Story Mode (workflow and approach-comparison visuals)
- Animal Journey Cards and contrast views
- Overview of current generated results
- Model comparison figures and tables
- Hypothesis Lab (H1/H3/H5 + H2/H4 descriptive checks)
- Reliability diagnostics, SHAP interpretability, risk explorer, campaign finder, adoption timeline
- Trust & Limits (evidence-pack view with subgroup selector, calibration-gap chart, model-struggle table)
- Generated artifact manifest and thesis/methodology report reader
- Model sensitivity form (shows how the trained model responds to different records; uses association language, not causal counterfactuals)

---

## Technology Stack

| Library | Purpose |
|---------|---------|
| **Python** | Main language |
| **pandas, numpy** | Data wrangling, feature engineering, aggregation, metrics |
| **scikit-learn** | Splitting, baselines, random forests, histogram boosting, preprocessing, evaluation |
| **CatBoost** | Advanced classifier/regressor for categorical-heavy shelter data |
| **SHAP** | Local and global model explanation (predictive association, not causality) |
| **Streamlit** | Thesis demo dashboard |
| **Altair, Plotly** | Interactive and publication-friendly dashboard visualizations |
| **Matplotlib** | Saved report figures |
| **joblib** | Model artifact serialization |
| **pytest** | Automated pipeline validation |

---

## Schema Compatibility Rules

Some generated report tables intentionally include **alias columns** for acceptance and dashboard compatibility. These are additive — existing readers can continue using original internal column names.

| Table | Internal column | Alias |
|-------|----------------|-------|
| H3 adopted-only | `group_value`, `all_records` | `age_group`, `records` |
| Model selection | `animal_subset` | `subset` |
| Threshold table | `threshold_label` | `threshold_name` |
| Calibration | internal columns | `subset`, `records` |
| Reliability red flags | `cohort`, `value` | `subgroup_field`, `subgroup_value`, `mean_predicted_probability` |
| Leakage audit | internal columns | `role`, `allowed_as_feature`, `leakage_risk`, `notes` |

---

## Artifact Manifest

`reports/artifact_manifest.csv` is a lightweight tracked manifest of generated thesis artifacts. Large raw data, processed data, and model binaries remain local/generated assets (gitignored).

Status values: `present` / `missing`

Manifest text is normalized to avoid mojibake dash artifacts in generated Markdown and CSV cells. The manifest is displayed in the Streamlit Artifacts tab.

---

## Test Suite

Configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

**Coverage areas:**

- Data loading, cleaning, matching, dataset build
- Feature engineering and feature-set leakage checks
- Split logic and model artifact helpers
- Baseline, boosting, and CatBoost output contracts
- EDA, analysis, diagnostics, and report outputs
- Evidence pack and subgroup reliability functions
- Data audit, leakage audit, target definitions, and artifact manifest
- Dashboard data helpers and story content
- Acceptance/schema aliases for generated artifacts
- Local explanation example generation and manifest registration

**Run tests:**

```bash
pytest
```

Known warning: pandas may emit a `FutureWarning` from evidence-pack concatenation when synthetic tests use empty/all-NA frames. This does not currently fail the suite.

---

## Thesis Chapter Mapping

| Thesis chapter | Primary code location | Key artifacts |
|---------------|----------------------|--------------|
| Ch 3: Data and Preprocessing | `src/aac_adoption/data/` | `data/processed/modeling_dataset.csv`, data audit tables |
| Ch 4: Methodology | `src/aac_adoption/features/`, `src/aac_adoption/models/` | `feature_columns.json`, model artifacts |
| Ch 5: Results | `src/aac_adoption/analysis/`, `src/aac_adoption/diagnostics/` | model comparison tables, hypothesis figures, SHAP outputs |
| Ch 6: System Architecture | `scripts/`, `tests/`, `src/aac_adoption/` | `artifact_manifest.csv`, full pipeline runner |
| Ch 7: Discussion | `src/aac_adoption/reporting/` | evidence pack, subgroup reliability, model failure modes |

---

## What Is Not Implemented (By Design)

- **Docker / DVC / MLflow** — deferred; the pipeline is reproducible via scripts and `requirements.txt` without them.
- **Production API** — the dashboard is a thesis demo, not a production shelter system.
- **Survival models as primary framework** — KM curves are descriptive only; Cox/AFT is future work.
- **Hyperparameter tuning** — documented as planned improvement in [`ROADMAP.md`](ROADMAP.md).
