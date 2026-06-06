# Found Location Plan

## Summary

`Found Location` is useful, but not as raw text. In AAC intake data it behaves like a messy location string with several repeated patterns:

- city labels such as `Austin (TX)`
- county / region labels such as `Travis (TX)`
- explicit `Outside Jurisdiction`
- address-like strings
- intersections, highways, roads, airport references, and other place descriptors

The right first move is a **deterministic coarse taxonomy**, not NLP embeddings and not full geocoding.

## Exploratory Findings

From `data/raw/intakes.csv`:

- rows: `173,812`
- nonempty `Found Location` values: `173,812`
- unique values: `70,183`

Top exact values:

- `Austin (TX)` appears `31,541` times
- `Travis (TX)` appears `3,796` times
- `Outside Jurisdiction` appears `2,083` times

Pattern scan:

- `contains_austin`: about `81.95%`
- `contains_tx`: about `98.80%`
- `outside_jurisdiction`: about `1.20%`
- `contains_zip5`: about `11.87%`
- `contains_number_start`: about `59.57%`
- `airport` token: about `0.63%`
- `intersection-ish` patterns using `and`, `/`, `&`, or `intersection`: common
- `highway / road / lane / street / drive / boulevard` references are also common

Examples found:

- `Airport And Denson in Austin (TX)`
- `Airport/46Th St in Austin (TX)`
- `Berkman Dr & Briarcliff Blvd in Austin (TX)`
- `Street / road / highway` variants
- `3600 Presidential Blvd (Airport) in Austin (TX)`
- `I35 And 71 Intersection in Austin (TX)`
- `812 Intersection Of 183 in Travis (TX)`

This field is therefore **semi-structured location data**, not open-ended language.

## Recommended Taxonomy

Use a small set of derived fields:

- `found_location_kind`
  - `austin_city`
  - `county_or_region`
  - `outside_jurisdiction`
  - `intersection`
  - `address_like`
  - `other`
- `found_location_area`
  - extracted city / county token when stable and clear
- optional flags
  - `is_austin_found_location`
  - `is_outside_jurisdiction`
  - `is_intersection_location`
  - `is_address_like_location`
  - `is_airport_location`

Suggested rule order:

1. `Outside Jurisdiction`
2. explicit intersection markers: `intersection`, `/`, `&`, ` and `
3. address-like strings starting with a number
4. airport-specific marker
5. trailing `in X (TX)` extraction
6. fallback `other`

## Why This Is Better Than NLP

- high interpretability
- no geocoder dependency
- low maintenance
- stable across runs
- useful for thesis tables and model features

Raw NLP on this field would add complexity without clear value because the strings are mostly short place descriptors, not narrative text.

## Research Use

Best use cases:

- descriptive breakdowns of adoption / outcome by location bucket
- counts by Austin vs outside Austin
- intersection vs address-like patterns
- airport / highway corridor analysis if it proves meaningful

Not recommended:

- raw string as a feature
- embeddings
- full free-text language processing

## Implementation Intention

When implemented, the pipeline should:

- preserve the original `Found Location` string in raw data
- derive coarse location buckets during cleaning or feature engineering
- keep the derived fields intake-time only
- document the choice in README and thesis support docs
- add tests for representative strings and weird edge cases

## Implementation Tasks

- [x] Keep raw `found_location` available long enough for intake-time feature engineering.
- [x] Add deterministic parser for `found_location_kind`, `found_location_area`, and location flags.
- [x] Register only derived coarse fields as model features; keep raw `found_location` out of model feature metadata.
- [x] Add tests for exact city/county labels, outside jurisdiction, intersections, address-like strings, airport references, missing values, and feature metadata.
- [x] Document ML-safe assumptions near the implementation plan.

## Saved Implementation Knowledge

- Loader converts `Found Location` to `found_location` via snake_case normalization.
- `build_modeling_dataset` drops unlisted intake columns before matching, so `found_location` must be in `INTAKE_COLUMNS_TO_KEEP` before features can be derived.
- Derived location fields are intake-time only because they come from the intake record and are created before outcome target creation.
- Raw `found_location` is intentionally not included in `BASE_INTAKE_TIME_FEATURES`, so `feature_columns.json` cannot use high-cardinality raw location text.
- Baseline and sklearn boosting models need the derived categorical/boolean location features in `CATEGORICAL_FEATURES`; CatBoost reuses that list.
- No geocoding, embeddings, external lookups, or target-aware encoding are used; rules are deterministic and reproducible.

## Assumption

Default assumption: this field stays **descriptive plus coarse-feature** first, not geospatially exact. If later work shows location is predictive enough to justify geocoding, that can be a separate phase.
