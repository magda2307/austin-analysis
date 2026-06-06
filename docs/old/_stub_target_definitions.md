# Target Variable Definitions — AAC Adoption ML Pipeline

## Purpose

This document is the single authoritative reference for every target variable in the pipeline.
All thesis text, dashboard labels, report outputs, and code must use consistent naming and semantics as defined here.
This document is designed for machine reading by AI agents and for human reviewers checking terminology consistency.

---

## Target Taxonomy

### 1. Classification Target

| Property | Value |
|---|---|
| Column name | `classification_target` |
| Alias columns | `adopted`, `is_adopted`, `target_adopted` |
| Data type | int (0 or 1) |
| Defined as | `1` if `outcome_type == "Adoption"` (case-insensitive strip), else `0` |
| Scope | All matched intake/outcome episodes (dogs and cats) |
| Used for | Binary classification: predict whether the matched outcome episode ended in adoption |
| Allowed thesis labels | "adoption indicator", "adopted vs. not adopted", "binary adoption target" |
| Forbidden labels | "adoption speed target", "timing target" |

**Interpretation note:**
This column predicts whether the animal's matched outcome was an adoption. It does not predict when. Non-adoption outcomes include Transfer, Return to Owner, Euthanasia, and Died. The model trained on this target learns the *probability* of adoption, not the *speed* of adoption.

**Code location:** `src/aac_adoption/data/build_dataset.py`, `build_modeling_dataset()` function.

```python
# Exact definition in pipeline
outcome_type_normalized = dataset["outcome_type"].fillna("").astype(str).str.strip().str.lower()
dataset["adopted"] = outcome_type_normalized.eq("adoption")
dataset["classification_target"] = dataset["adopted"].astype(int)
```

---

### 2. Regression Target — Primary (Length of Stay / Days to Matched Outcome)

| Property | Value |
|---|---|
| Column name | `regression_target_days` |
| Alias columns | `days_to_outcome`, `length_of_stay` |
| Data type | float (non-negative) |
| Defined as | `(outcome_datetime - intake_datetime).total_seconds() / 86400` |
| Scope | All matched intake/outcome episodes (dogs and cats) |
| Used for | Regression: predict how many days an animal remained in care until its matched outcome |
| Allowed thesis labels | "days to outcome", "length of stay", "predicted days to outcome", "predicted length of stay", "time to matched outcome" |
| Forbidden labels | "adoption speed", "days to adoption", "predicted wait until adoption" (unless the subset is adopted animals only — see Target 3) |

**Interpretation note:**
This is an operational length-of-stay prediction. The outcome is the animal's *next matched outcome record*, which may be adoption, transfer, return-to-owner, or another disposition. It is NOT guaranteed to be adoption. Describing this as "predicted time to adoption" is methodologically wrong unless the prediction is restricted to adopted animals only.

Shelters care about length of stay regardless of outcome type because all occupied kennels have holding costs. This is why the regression target is defined over all outcomes, not just adoptions.

**Code location:** `src/aac_adoption/data/build_dataset.py`, `build_modeling_dataset()` function.

```python
# Exact definition in pipeline
dataset["days_to_outcome"] = (
    dataset["outcome_datetime"] - dataset["intake_datetime"]
).dt.total_seconds() / 86400
dataset["regression_target_days"] = dataset["days_to_outcome"]
dataset["length_of_stay"] = dataset["days_to_outcome"]
```

**Regression MAE label in all reports:**
> "combined regression MAE: X days for length-of-stay / days-to-outcome prediction"

---

### 3. Adoption-Only Timing Target — Descriptive / Optional

| Property | Value |
|---|---|
| Column name | `days_to_adoption` |
| Data type | float (non-negative) or NaN |
| Defined as | `days_to_outcome` where `outcome_type == "Adoption"`, else `NaN` |
| Scope | Adopted animals only (a subset of all matched episodes) |
| Used for | Descriptive adoption-speed analysis; optional adopted-only regression |
| Allowed thesis labels | "days to adoption (among adopted animals)", "adoption timing (adopted subset)", "median days to adoption" |
| Forbidden labels | Using this column as the main regression target without clearly stating the adopted-only scope |

**Interpretation note:**
This column exists to support H3 (age and adoption timing) without confusing non-adoption outcomes with adoption speed. When a thesis statement says "young animals are adopted in X median days", it should use this column or explicitly filter to `outcome_type == "Adoption"`. The main regression model uses `regression_target_days` (all outcomes) for operational LOS prediction.

**Code location:** `src/aac_adoption/data/build_dataset.py`, `build_modeling_dataset()` function.

```python
# Exact definition in pipeline
dataset["days_to_adoption"] = np.where(
    dataset["adopted"],
    dataset["days_to_outcome"],
    np.nan,
)
```

**Generated artifact:** `reports/tables/h3_adopted_only_age_speed.csv`

---

### 4. Survival / Time-to-Event Target — Future Work

| Property | Value |
|---|---|
| Column name | Not yet defined |
| Framework | Kaplan–Meier / Cox proportional hazards (not implemented as main model) |
| Censoring | Intakes without a future outcome (cannot determine final outcome) |
| Competing risks | Transfer, Euthanasia, Return-to-Owner all "compete" with Adoption |
| Status | **Not the main modeling framework.** Discussed as future work. |

**What IS implemented (descriptive only):**
- Empirical adoption survival curves by `animal_type`, `age_group`, `covid_period`, and `intake_type`.
- Kaplan–Meier descriptive curves using `lifelines` library.
- These are descriptive time-to-adoption views for adopted animals, NOT a full survival model.

**Artifacts generated (descriptive only):**
- `reports/tables/adoption_survival_curves.csv`
- `reports/figures/km_adoption_by_animal_type.png`
- `reports/figures/km_adoption_by_age_group.png`
- `reports/figures/km_adoption_by_covid_period.png`
- `reports/summary/survival_descriptive_note.md`

**Thesis framing:**
> "These curves are descriptive time-to-adoption views among adopted animals. They are not the main modeling framework and do not replace the supervised ML comparison. Full time-to-event survival modeling with censoring and competing risks is outside the scope of this thesis and is noted as a natural extension for future work."

---

## Leakage Control Summary

The following columns are **targets or outcome-derived metadata** and must never appear in `feature_columns.json` or be passed to any trained model as predictors:

| Column | Category | Reason |
|---|---|---|
| `classification_target` | target | binary adoption label |
| `regression_target_days` | target | days to outcome |
| `days_to_outcome` | target/alias | same as regression_target_days |
| `length_of_stay` | target/alias | same as days_to_outcome |
| `days_to_adoption` | target/alias | adopted-only timing |
| `adopted` | target/alias | binary adoption flag |
| `is_adopted` | target/alias | binary adoption flag |
| `target_adopted` | target/alias | binary adoption flag |
| `outcome_type` | metadata | post-intake outcome label |
| `outcome_subtype` | metadata | post-intake outcome detail |
| `outcome_datetime` | metadata | post-intake timestamp |
| `sex_upon_outcome` | metadata | post-intake measurement |
| `age_upon_outcome` | metadata | post-intake measurement |

**Validation:** `src/aac_adoption/features/feature_sets.py` — `LEAKAGE_COLUMNS` set and `validate_no_leakage()` function.
**Leakage audit script:** `scripts/generate_leakage_audit.py`

---

## Consistent Label Rules for All Files

| Context | Correct label | Incorrect label |
|---|---|---|
| Regression model output (all animals) | "predicted days to outcome" / "predicted length of stay" | "predicted adoption speed", "days to adoption" |
| Regression MAE in reports | "MAE X days for length-of-stay prediction" | "adoption speed error" |
| H3 table (all animals) | "median days to outcome by age group" | "adoption speed by age" |
| H3 adopted-only table | "median days to adoption (adopted animals only)" | "adoption speed" without qualification |
| Dashboard regression metric card | "Predicted days to outcome" | "Predicted days to adoption" |
| SHAP regression interpretation | "associated with predicted days to outcome" | "causes faster adoption" |

---

## Relationship Between Targets

```
All matched intake/outcome episodes (N = 162,390)
├── classification_target=1 (adopted, ~52%)  ──► days_to_adoption = days_to_outcome
│                                                  ↑ Used for adopted-only descriptive speed analysis
└── classification_target=0 (not adopted, ~48%) ──► days_to_adoption = NaN

regression_target_days = days_to_outcome (ALL episodes, regardless of outcome type)
    ↑ Main regression target: predicts length of stay, not adoption speed
```
