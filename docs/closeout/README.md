# Thesis Closeout Control Center

This folder is the execution source of truth for finishing the AAC adoption
project. It supplements `docs/PROJECT_CLOSEOUT_TASKS.md` with implementation
ordering, exact validation gates, and the manual long-run procedure.

## Locked Decisions

1. Main classification target remains adoption vs. non-adoption for matched
   intake/outcome episodes.
2. Main regression target remains days to any matched outcome.
3. Adopted-only timing remains a separate descriptive/optional analysis.
4. Full Cox and competing-risk modeling is outside thesis acceptance. Keep only
   descriptive Kaplan-Meier evidence that is methodologically valid.
5. Base models train on 2013-2021, calibration fits on 2022, selection and
   threshold choice use 2023, and test data is 2024-2025 frozen final evaluation.
6. Random-split fallback may support development fixtures, but it must never be
   labeled thesis evaluation or enter final model selection.
7. Dashboard errors must be explicit. No default probability or LOS value may be
   displayed as a prediction.
8. Quick and smoke validation must write to temporary directories. Only the
   explicit manual regeneration phase may replace canonical thesis artifacts.
9. Horizon classifiers use a separate all-intake cohort, not the matched-only
   classification/regression dataset.
10. Canonical regeneration requires a clean committed source tree. Every producer
    writes a receipt; final documentation is completed before the manifest.

## Documents

- `2026-06-08-thesis-closeout-implementation-plan.md`: dependency-ordered,
  subagent-ready implementation plan.
- `validation-matrix.md`: required tests and evidence by task and acceptance tier.
- `manual-regeneration-runbook.md`: long commands the user will run after code
  gates pass.

## Execution Rule

Complete tasks in plan order. Do not regenerate downstream artifacts while an
upstream producer is failing. A task is not complete because its unit test passes;
its listed contract tests and downstream schema checks must also pass.
