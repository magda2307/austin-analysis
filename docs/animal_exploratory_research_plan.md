# Animal-Centered Exploratory Research Plan

This plan extends the thesis dashboard with animal-first evidence and story views. The goal is to make the project feel less like a generic modeling exercise and more like an analysis of real shelter animal journeys.

## Focus Questions

- Which animal profiles have the strongest or weakest adoption outcomes?
- How do pit-bull-type dogs compare with other dog breed groups?
- How do black/dark cats compare with other cats?
- How do domestic-cat breed labels compare with less common cat breed groups?
- How do senior animals compare with baby animals?
- How do named animals compare with unnamed animals?
- Which health-condition profiles have lower adoption or longer waits?
- Which behavior-support proxy groups appear in the data?
- Which cohorts need visibility or early support?
- Where do model predictions align with similar historical animals?

## Core Outputs

Generate:

- `animal_archetypes.csv`
- `vulnerable_profiles.csv`
- `profile_model_error.csv`
- `profile_contrasts.csv`
- `health_behavior_profiles.csv`

Add figures:

- top archetypes by records,
- vulnerable animal profiles,
- key contrasts:
  - pit-bull-type dogs vs other dog groups,
  - black/dark cats vs other cats,
  - domestic-cat labels vs other cat breed groups,
  - senior vs baby,
  - named vs unnamed by species.

## Animal Journey Card

Each card should show:

- species,
- age group,
- intake type,
- condition,
- health profile,
- behavior-support flag,
- breed group,
- color group,
- predicted adoption chance,
- predicted wait,
- similar-case history,
- top SHAP-associated reasons,
- visibility need label.

Prediction fields come from existing model artifacts and diagnostics when available. Descriptive fields come from intake-time and outcome-summary data. SHAP fields should always be phrased as model associations, not causes.

## Visibility Need Labels

Use transparent heuristic labels:

- `quick placement likely`
- `needs visibility`
- `long-stay risk`
- `outcome support priority`

These labels are decision-support signals, not causal claims.

## Health and Personality Proxies

The dataset does not contain a reliable full personality or temperament profile. The defensible version is:

- use `intake_condition` for health and vulnerability groups,
- use `outcome_subtype` values such as behavior, aggressive, rabies risk, or court/investigation as behavior-support signals,
- avoid claiming breed or color measures personality,
- describe these columns as shelter-record proxies for care context.

This still gives the thesis an animal-centered lens without turning noisy administrative fields into causal or moral labels.

## Dashboard

Add `Animal Stories` tab with:

- archetype selector,
- journey card,
- representative CatBoost prediction for the selected card,
- local CatBoost SHAP reasons when model artifacts are available,
- model-wide SHAP fallback reasons when only generated SHAP tables are available,
- similar historical cases with outcome mix,
- key contrast cards,
- vulnerable profiles table,
- health and behavior-support profile table.

## Journey Cards v2 Implementation

The implemented v2 card converts each selected archetype into a representative intake-time model record. That record is used to show:

- predicted adoption probability,
- predicted days to outcome,
- prediction-derived visibility label,
- exact/coarse similar historical cases,
- adoption, transfer, return-to-owner, and euthanasia mix for similar cases,
- local CatBoost SHAP features for the representative record.

If model artifacts or SHAP dependencies are unavailable, the dashboard falls back to artifact-driven explanation tables and clear setup messages.

## Current Implemented Command

```bash
python scripts/generate_animal_research.py --data data/processed/modeling_dataset.csv
```

Implemented tables:

- `reports/tables/animal_archetypes.csv`
- `reports/tables/vulnerable_profiles.csv`
- `reports/tables/profile_contrasts.csv`
- `reports/tables/profile_model_error.csv`
- `reports/tables/health_behavior_profiles.csv`

Implemented figures:

- `reports/figures/animal_archetypes_top.png`
- `reports/figures/vulnerable_profiles.png`
- `reports/figures/profile_contrasts_adoption_rate.png`
