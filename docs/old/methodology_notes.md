# Methodology Notes — AAC Adoption ML Pipeline

## Purpose

This document records key methodological design decisions for the thesis pipeline.
It is written in English for use by AI agents and thesis reviewers.
It explains *why* certain approaches were chosen and how to defend those choices.

---

## Why Regression Is Still Useful (Task 2.4 Justification)

### The Challenge

A common reviewer objection to using regression (predicting `days_to_outcome`) is:

> "The target mixes animals that were adopted with animals that were transferred or euthanized. The predicted value is not comparable across animals because different outcome types have different typical durations."

This objection is valid but does not invalidate the regression task. Here is the full defense:

### Core Argument

**1. Shelters care about length of stay, not just whether adoption happened.**

Austin Animal Center manages ~18,000+ animals per year. Every day an animal occupies a kennel has a cost: feeding, veterinary monitoring, staff time, and space that another incoming animal cannot use. Whether the animal is ultimately adopted, transferred, or returned to an owner, the shelter must plan for its housing duration. `days_to_outcome` is therefore an operationally meaningful prediction target regardless of outcome type.

**2. `days_to_outcome` is not equivalent to "adoption speed".**

Adoption speed would require restricting the dataset to animals that were ultimately adopted. The main regression model predicts *length of stay until any outcome*, which is the operational quantity a shelter scheduler cares about. The thesis explicitly separates:
- Main regression: `regression_target_days` = `days_to_outcome` over all episodes
- Adopted-only timing: `days_to_adoption` restricted to `outcome_type == "Adoption"` — used only for descriptive H3 analysis

**3. Regression MAE has direct operational meaning.**

A MAE of 18.55 days means the model's length-of-stay prediction is off by about 18 days on average. A shelter director can interpret this directly: "If the model says an animal will leave in 10 days, it might actually stay between ~0 and ~36 days." This is an honest, operationally grounded error framing.

**4. Dummy median baseline is meaningful.**

Because length-of-stay is right-skewed (most animals leave quickly, some stay very long), a dummy regressor that always predicts the median is a strong baseline. The improvement over this baseline demonstrates that intake-time features provide real predictive signal for length-of-stay, even if the MAE appears large in absolute terms.

**5. The regression task supports resource-allocation use cases beyond adoption.**

Even non-adopted animals need placement: transfers to rescue groups, return-to-owner coordination, and euthanasia decisions all benefit from length-of-stay estimates. Restricting the regression to adopted animals only would make the model less useful for the full shelter workflow.

### Summary Statement for Thesis

> The regression target `regression_target_days` predicts length of stay until any matched outcome. This is operationally equivalent to "time the animal occupies a kennel" rather than "time until adoption." Adoption-only timing is analyzed separately in the descriptive H3 section using the `days_to_adoption` variable. Full time-to-event survival modeling with censoring and competing risks would be a natural extension but is outside the main scope of this thesis.

---

## Why Not Survival Analysis as the Main Model?

### The Objection

A reviewer may ask: "Why did you use supervised regression instead of Kaplan–Meier or Cox proportional hazards models, which are the standard approach for time-to-event data?"

### Response

**1. Most animals in this dataset have a resolved outcome.**

Survival analysis is most important when a large fraction of observations are *censored* (outcome not yet observed). In the AAC dataset, the pipeline discards intakes without a matched future outcome. The final modeling dataset contains only resolved episodes. The censoring problem is therefore much smaller than in typical survival analysis settings (e.g., clinical trials where patients are lost to follow-up).

**2. The regression target is not time-to-adoption; it is time-to-any-outcome.**

If the target were "time to adoption" with other outcomes treated as competing events, a formal competing-risks survival model would be correct. In this pipeline, `days_to_outcome` is the generic length of stay until any disposition, so standard regression with an appropriate loss function (MAE) is a defensible choice.

**3. Descriptive survival curves are provided.**

Kaplan–Meier-style empirical curves are generated for adopted animals grouped by `animal_type`, `age_group`, `covid_period`, and `intake_type`. These serve as descriptive evidence for H3 (age and adoption timing) without requiring a full survival model.

**4. Regression is more interpretable for the thesis use case.**

The thesis goal includes a dashboard where a shelter worker can see "predicted days to outcome: 12 days." A hazard ratio from a Cox model is harder to explain to a non-statistical audience. The ML regression output is more directly actionable.

**5. Future work note.**

Full survival modeling with:
- Censoring for intakes that have no observed future outcome
- Competing risks (adoption vs. transfer vs. euthanasia vs. return-to-owner)
- Time-varying covariates (e.g., condition changes during stay)

would be a natural and scientifically rigorous extension of this thesis. The descriptive KM curves generated in this pipeline provide the foundation for that future work.

### Summary Statement for Thesis

> Kaplan–Meier descriptive curves are generated as a supplementary descriptive layer. They show empirical adoption probabilities over time, grouped by animal type, age, and period. These curves are not the main modeling framework. Full survival analysis with censoring and competing risks is noted as future work. The main supervised ML models (classification and regression) were chosen for their interpretability, direct operational output format, and suitability for the resolved-episode structure of the AAC dataset.

---

## Predictive Association vs. Causal Claims

### What the Models Do and Do Not Show

The ML pipeline produces **predictive associations**, not causal evidence. All interpretation outputs must use this language.

**Allowed:**
- "X is associated with higher predicted adoption probability"
- "X is linked to model output"
- "intake-time predictors are associated with length-of-stay patterns"
- "the model identifies age as a strong predictor of outcome timing"

**Forbidden:**
- "X causes faster adoption"
- "COVID caused adoption rates to rise"
- "black animals are discriminated against" (without framing as descriptive association only)
- "reducing X would improve outcomes"

**Why this matters:**
The dataset is observational. There is no randomized assignment of any feature. Confounders are unknown. The ML models optimize prediction accuracy, not causal identification. Any causal framing would require a different methodology (natural experiments, difference-in-differences, instrumental variables, etc.).

The COVID-period variable (`covid_period`) is an intake-date label. Differences across periods are *associated with* model output but could reflect policy changes, population changes, media effects, or unobserved confounders. The thesis must not say "COVID caused adoption to increase."

The `is_black_or_dark` variable is a color descriptor. The model's association between dark coloring and lower predicted adoption probability reflects patterns in the data but does not prove discrimination or causal bias. Framing must be: "Dark-colored animals show lower observed adoption rates, a pattern that is descriptively consistent with the so-called black dog syndrome reported in shelter literature."

---

## Intake-Time Feature Set Justification

All model features are available at the moment of intake (before any outcome is known). This is a fundamental leakage-safety guarantee. The features fall into these families:

| Family | Features | Purpose |
|---|---|---|
| Animal identity | `animal_type`, `breed`, `primary_breed`, `simplified_breed_group` | Species and appearance |
| Appearance | `color`, `primary_color`, `simplified_color_group`, `is_black_or_dark` | Color-related H4 evidence |
| Name status | `is_named` | Proxy for prior ownership; duplicate `has_name` alias is excluded from model features |
| Age | `age_upon_intake`, `age_days`, `age_group` | H3 evidence; linear age aliases are excluded from model features |
| Intake circumstances | `intake_type`, `intake_condition` | H1 evidence |
| Timing | `intake_year`, `intake_month`, `covid_period` | H2, H5 evidence; redundant calendar aliases are kept for reports only |
| Location | `found_location_kind`, `found_location_area`, `is_austin_found_location`, etc. | Intake context |
| Sex | `sex_upon_intake` | Biological and reproductive status at intake |

Context features (optional, intake-date-based):
| Feature | Source | Type |
|---|---|---|
| `daily_temp_max`, `daily_temp_min`, `daily_precipitation` | Austin weather | Intake-date weather |
| `is_extreme_heat`, `is_rainy_day` | Derived from weather | Environmental signal |
| `animal_311_requests_7d`, `animal_311_requests_30d` | Austin 311 | Prior animal service demand |
| `intake_volume_7d`, `intake_volume_30d` | Shelter intakes | Prior shelter pressure |

All context features use only dates **before** the intake date (rolling window does not include intake day itself), maintaining temporal integrity.

---

## Dataset Matching Logic

Each intake episode is matched to the nearest unused future outcome for the same animal. This is a **greedy nearest-future-match** algorithm:

1. Sort outcomes by datetime for each animal.
2. For each intake (sorted by intake_datetime), skip outcomes that occurred before the intake.
3. Assign the next available outcome to this intake.
4. Mark that outcome as used (cannot be reused for a later intake).

**Consequence:**
- An animal with N intakes and N outcomes gets N episode rows.
- An animal with N intakes and fewer outcomes loses the trailing intakes (counted as `unmatched_intakes`).
- No outcome is shared between two intake episodes.
- No negative `days_to_outcome` values are possible (validated by `validate_modeling_dataset()`).

This is methodologically equivalent to treating each shelter stay as a separate operational episode, which is the correct unit of analysis for shelter resource planning.

**Limitation:**
If an animal was transferred between shelters and readmitted, the re-admission creates a new episode that is independent of the previous episode. This is intentional and correct behavior — each stay is a separate resource-planning problem.
