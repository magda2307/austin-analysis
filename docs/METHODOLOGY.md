# Methodology — AAC Adoption ML Pipeline

This document is the authoritative reference for methodological framing, target variable definitions, causal language rules, feature set justification, and dataset matching logic. All thesis text, dashboard labels, report outputs, and code must be consistent with this document.

---

## Scope and Claims

### What this project claims

This pipeline demonstrates **predictive associations** between intake-time features and adoption outcomes / length of stay in Austin Animal Center records (2013–2025).

### What this project does NOT claim

- That intake features *cause* adoption outcomes.
- That COVID *caused* adoption rates to change.
- That dark-colored animals are *discriminated against* (descriptive association only).
- That the regression output equals *adoption speed* — it is length of stay until any matched outcome.

### Required terminology in all outputs

| Use | Avoid |
|-----|-------|
| `predictive association`, `associated with` | `causes adoption`, `proves` |
| `linked to model output` | `COVID caused...` |
| `intake-time predictors` | `adoption speed` (unless subset is adopted animals only) |
| `length of stay`, `time to outcome` | `days to adoption` (unless explicitly filtered) |
| `descriptive time-to-adoption evidence` | `reduces adoption time` |

---

## Target Variable Definitions

### 1. Classification Target

| Property | Value |
|----------|-------|
| Column name | `classification_target` |
| Alias columns | `adopted`, `is_adopted`, `target_adopted` |
| Data type | int (0 or 1) |
| Defined as | `1` if `outcome_type == "Adoption"` (case-insensitive), else `0` |
| Scope | All matched intake/outcome episodes (dogs and cats) |
| Allowed thesis labels | "adoption indicator", "adopted vs. not adopted", "binary adoption target" |
| Forbidden labels | "adoption speed target", "timing target" |

```python
outcome_type_normalized = dataset["outcome_type"].fillna("").astype(str).str.strip().str.lower()
dataset["adopted"] = outcome_type_normalized.eq("adoption")
dataset["classification_target"] = dataset["adopted"].astype(int)
```

### 2. Regression Target — Primary (Length of Stay)

| Property | Value |
|----------|-------|
| Column name | `regression_target_days` |
| Alias columns | `days_to_outcome`, `length_of_stay` |
| Data type | float (non-negative) |
| Defined as | `(outcome_datetime - intake_datetime).total_seconds() / 86400` |
| Scope | All matched intake/outcome episodes |
| Allowed thesis labels | "days to outcome", "length of stay", "predicted days to outcome" |
| Forbidden labels | "adoption speed", "days to adoption", "predicted wait until adoption" (unless subset is adopted only) |

**Interpretation:** This is an operational length-of-stay prediction. The matched outcome may be adoption, transfer, return-to-owner, or another disposition. Describing this as "predicted time to adoption" is methodologically wrong unless explicitly filtered to adopted animals.

```python
dataset["days_to_outcome"] = (
    dataset["outcome_datetime"] - dataset["intake_datetime"]
).dt.total_seconds() / 86400
dataset["regression_target_days"] = dataset["days_to_outcome"]
```

### 3. Adoption-Only Timing Target — Descriptive / H3 Only

| Property | Value |
|----------|-------|
| Column name | `days_to_adoption` |
| Data type | float or NaN |
| Defined as | `days_to_outcome` where `outcome_type == "Adoption"`, else `NaN` |
| Scope | Adopted animals only |
| Allowed thesis labels | "days to adoption (among adopted animals)", "adoption timing (adopted subset)" |
| Forbidden | Using as the main regression target without stating adopted-only scope |

```python
dataset["days_to_adoption"] = np.where(
    dataset["adopted"], dataset["days_to_outcome"], np.nan
)
```

### 4. Survival / Time-to-Event — Future Work

Kaplan–Meier descriptive curves are generated for adopted animals grouped by `animal_type`, `age_group`, `covid_period`, and `intake_type`. These serve as **descriptive evidence** for H3 only. Full survival modeling with censoring and competing risks is outside the main scope of this thesis.

### Target Relationship Diagram

```
All matched episodes (N ≈ 162,390)
├── classification_target=1 (adopted, ~52%) ──► days_to_adoption = days_to_outcome
│                                                 ↑ Used for H3 adopted-only timing
└── classification_target=0 (not adopted, ~48%) ──► days_to_adoption = NaN

regression_target_days = days_to_outcome (ALL episodes, regardless of outcome type)
    ↑ Main regression target: predicts length of stay, not adoption speed
```

---

## Leakage Control

The following columns are **targets or outcome-derived metadata** and must never appear in `feature_columns.json` or be passed to any model as predictors:

| Column | Category | Reason |
|--------|----------|--------|
| `classification_target` | target | binary adoption label |
| `regression_target_days` | target | days to outcome |
| `days_to_outcome` | target/alias | same as regression_target_days |
| `length_of_stay` | target/alias | same as days_to_outcome |
| `days_to_adoption` | target/alias | adopted-only timing |
| `adopted`, `is_adopted`, `target_adopted` | target/alias | binary adoption flags |
| `outcome_type` | metadata | post-intake outcome label |
| `outcome_subtype` | metadata | post-intake outcome detail |
| `outcome_datetime` | metadata | post-intake timestamp |
| `sex_upon_outcome` | metadata | post-intake measurement |
| `age_upon_outcome` | metadata | post-intake measurement |

**Validation:** `src/aac_adoption/features/feature_sets.py` — `LEAKAGE_COLUMNS` set and `validate_no_leakage()` function.  
**Leakage audit:** `scripts/generate_leakage_audit.py`

---

## Intake-Time Feature Set

All model features are available at the moment of intake (before any outcome is known). This is the fundamental leakage-safety guarantee.

| Family | Features | Hypothesis |
|--------|----------|-----------|
| Animal identity | `animal_type`, `breed`, `primary_breed`, `simplified_breed_group` | H1 |
| Appearance | `color`, `primary_color`, `simplified_color_group`, `is_black_or_dark` | H4 |
| Name status | `has_name`, `is_named` | — |
| Age | `age_upon_intake`, `age_days`, `age_months`, `age_years`, `age_group` | H3 |
| Intake circumstances | `intake_type`, `intake_condition` | H1 |
| Timing | `intake_year`, `intake_month`, `intake_quarter`, `intake_season`, `covid_period` | H2, H5 |
| Location | `found_location_kind`, `found_location_area`, `is_austin_found_location`, flags | — |
| Sex | `sex_upon_intake` | — |

**Optional context features** (intake-date-based, prior window only):

| Feature | Source |
|---------|--------|
| `daily_temp_max`, `daily_temp_min`, `daily_precipitation` | Austin weather |
| `is_extreme_heat`, `is_rainy_day` | Derived from weather |
| `animal_311_requests_7d`, `animal_311_requests_30d` | Austin 311 |
| `intake_volume_7d`, `intake_volume_30d` | Shelter intakes |

All context features use only dates **before** the intake date. Rolling windows do not include the intake day itself.

**Note on feature redundancy:** `age_days`, `age_months`, and `age_years` are collinear; `has_name` and `is_named` are near-identical. These redundancies are documented in [`docs/ROADMAP.md`](ROADMAP.md) as a planned cleanup item.

---

## Dataset Matching Logic

Each intake episode is matched to the nearest unused future outcome for the same animal using a **greedy nearest-future-match** algorithm:

1. Sort outcomes by datetime for each animal.
2. For each intake (sorted by `intake_datetime`), skip outcomes that occurred before the intake.
3. Assign the next available outcome to this intake.
4. Mark that outcome as used (cannot be reused for a later intake).

**Consequences:**
- An animal with N intakes and N outcomes gets N episode rows.
- An animal with N intakes and fewer outcomes loses trailing intakes (counted as `unmatched_intakes`).
- No outcome is shared between two intake episodes.
- No negative `days_to_outcome` values are possible (validated by `validate_modeling_dataset()`).

**Limitation:** If an animal was transferred between shelters and readmitted, the re-admission creates a new independent episode. This is intentional — each stay is a separate resource-planning problem. Re-intake ambiguity detection (checking for intermediate intakes within a matched pair) is a planned improvement; see [`docs/ROADMAP.md`](ROADMAP.md).

---

## Why Regression Instead of Survival Analysis?

**Short answer:** Most animals in this dataset have resolved outcomes, so censoring is not the dominant concern. The regression target (`days_to_outcome`) is operationally meaningful and more interpretable for non-statistical audiences than hazard ratios.

**Full defense:**

1. **Shelters care about length of stay, not just whether adoption happened.** Every kennel-day has a cost regardless of outcome type.
2. **`days_to_outcome` ≠ "adoption speed".** Adoption-only timing is analyzed separately as `days_to_adoption` (H3 descriptive section).
3. **Regression MAE has direct operational meaning.** MAE ≈ 18 days means the model's LOS estimate is off by ~18 days on average.
4. **Descriptive survival curves are provided.** Kaplan–Meier curves for adopted animals by `animal_type`, `age_group`, and `covid_period` give a time-to-adoption view without requiring a full survival model.
5. **Full survival modeling is noted as future work.** It would add censoring for unresolved stays and competing-risks framing (adoption vs transfer vs euthanasia vs return-to-owner).

**Thesis statement:**
> *The regression target `regression_target_days` predicts length of stay until any matched outcome. This is operationally equivalent to "time the animal occupies a kennel" rather than "time until adoption." Adoption-only timing is analyzed separately in the descriptive H3 section using `days_to_adoption`. Full survival modeling with censoring and competing risks is noted as future work.*

---

## Predictive Association vs. Causal Claims

The ML pipeline produces **predictive associations**, not causal evidence. There is no randomized assignment of any feature. Confounders are unknown. ML models optimize prediction accuracy, not causal identification.

**Specific framing rules:**

- **COVID period:** `covid_period` is an intake-date label. Differences across periods are *associated with* model output but could reflect policy changes, population changes, media effects, or unobserved confounders.
- **Dark coloring:** `is_black_or_dark` is a color descriptor. Lower observed adoption rates for dark animals reflect patterns in the data but do not prove discrimination or causal bias. Correct framing: *"Dark-colored animals show lower observed adoption rates, a pattern descriptively consistent with the so-called black dog syndrome reported in shelter literature."*

---

## Consistent Label Rules

| Context | Correct | Incorrect |
|---------|---------|-----------|
| Regression model output (all animals) | "predicted days to outcome" / "predicted length of stay" | "predicted adoption speed", "days to adoption" |
| Regression MAE in reports | "MAE X days for length-of-stay prediction" | "adoption speed error" |
| H3 table (all animals) | "median days to outcome by age group" | "adoption speed by age" |
| H3 adopted-only table | "median days to adoption (adopted animals only)" | "adoption speed" without qualification |
| Dashboard regression metric | "Predicted days to outcome" | "Predicted wait until adoption" |
| SHAP regression interpretation | "associated with predicted days to outcome" | "causes faster adoption" |
