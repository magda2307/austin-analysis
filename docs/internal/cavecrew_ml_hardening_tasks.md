# Cavecrew ML Hardening Tasks

Date: 2026-06-06

Purpose: current cavecrew backlog only. Completed/stale review items live in `docs/internal/ml_hardening_done_reference.md`.

Ground rules:

- Run `git status --short` before edit.
- Do not revert unrelated user or agent work.
- One task per cavecrew-builder when possible.
- Use cavecrew-builder for 1-2 file fixes.
- Use cavecrew-investigator before broad matching/reporting work.
- Use cavecrew-reviewer after each diff.
- Builder final format:

```text
<path:line-range> -- <change <=10 words>.
verified: <command or re-read OK>.
```

- Reviewer final format:

```text
path:line: severity: problem. fix.
totals: N bug N risk N nit N q
```

## Task A - Verify Clean Calibration Stage

Use: `cavecrew-investigator`, then `cavecrew-reviewer`.

Priority: P0

Own files for investigation:

- `src/aac_adoption/models/calibrate.py`
- `src/aac_adoption/models/train_advanced.py`
- `scripts/calibrate_classifiers.py`
- `tests/test_calibration.py`
- `tests/test_calibration_advanced.py`

Problem:

- Calibration exists now, but roadmap/review status disagrees.
- Need prove no stale issues remain: Platt stays sigmoid, calibrator uses fitted model, validation is split between early stopping and calibration, calibrated metrics are not mixed with uncalibrated metrics.

Acceptance:

- `python scripts/calibrate_classifiers.py --help` exits 0.
- Tests prove Platt/sigmoid path is preserved.
- Advanced training emits coherent calibrated vs uncalibrated rows.
- Roadmap status can be updated from evidence, not optimism.

Suggested command:

```powershell
python scripts/calibrate_classifiers.py --help
python -m pytest tests/test_calibration.py tests/test_calibration_advanced.py tests/test_train_advanced_outputs.py -q
```

Review focus:

```text
src/aac_adoption/models/train_advanced.py:170: risk: calibrated and uncalibrated metrics may mix. Verify separate rows.
```

## Task B - Resolve LOS Target Leakage

Use: `cavecrew-investigator`, then `cavecrew-builder`.

Priority: P0

Own files:

- `src/aac_adoption/data/build_dataset.py`
- `src/aac_adoption/models/train_advanced.py`
- tests only if needed: `tests/test_build_dataset.py`

Problem:

- Original review flagged full-dataset target winsorization.
- Current dataset builder appears to preserve raw `days_to_outcome`, but need verify no target caps happen before split elsewhere.

Acceptance:

- No pre-split cap/winsor transform touches `length_of_stay`, `days_to_outcome`, or `regression_target_days`.
- If target capping exists, caps are fit on train only and stored in model metadata.
- Test proves extreme test-period target cannot change train target transform.

Investigator query:

```text
Find all winsor/cap/clip/log target transforms for LOS/regression targets. Return path:line only.
```

Commit:

```text
fix(regression): prevent target leakage
```

## Task C - Finish Censoring Summary

Use: `cavecrew-builder`.

Priority: P0

Own files:

- `src/aac_adoption/data/build_dataset.py`
- `scripts/generate_data_audit.py`
- `docs/ROADMAP.md`
- tests only if needed: `tests/test_horizon_targets.py`

Problem:

- Horizon columns now use `followup_days_available`.
- Roadmap still marks end-of-dataset censoring/follow-up summary incomplete.
- Need included/excluded row counts per horizon and cutoff date.

Acceptance:

- Artifact/table records rows included, censored, fast-adopted despite short follow-up, and excluded per 7/30/60/90 day horizon.
- Output includes extract/end date used for follow-up.
- Test verifies late-tail unresolved rows become `NaN` for unsafe horizons.
- `docs/ROADMAP.md` status updated only after generated output exists.

Commit:

```text
fix(targets): summarize horizon censoring
```

## Task D - Audit Re-Intake Ambiguity

Use: `cavecrew-investigator`, then `cavecrew-builder`.

Priority: P0

Own files:

- `src/aac_adoption/data/match_records.py`
- `scripts/audit_matching_ambiguity.py`
- tests only if needed: `tests/test_match_records.py`

Problem:

- Matching records re-intake metadata, but roadmap says ambiguous intake-outcome pairs are not rejected/summarized.

Acceptance:

- Detect same-animal intake between candidate intake and candidate outcome.
- Report clean, ambiguous, censored/unmatched, dropped counts.
- Decide and document whether ambiguous episodes are excluded or only flagged.
- Test includes animal with intake A, intake B, outcome C.

Commit:

```text
fix(data): audit reintake ambiguity
```

## Task E - Add Yearly Backtesting Artifact

Use: `cavecrew-builder`.

Priority: P1

Own files:

- `scripts/evaluate_backtesting.py`
- `scripts/run_full_pipeline.py`
- tests only if needed: new `tests/test_backtesting.py`

Problem:

- `scripts/evaluate_backtesting.py` exists, but roadmap says yearly backtesting table is still TODO.
- Need wire into reproducible artifact path and pipeline.

Acceptance:

- Backtesting writes stable CSV under `reports/tables/`.
- Pipeline can run the step or clearly skip with flag.
- Output includes train period, test year, subset, model, PR-AUC, ROC-AUC, Brier/ECE when available.
- Test verifies schema on tiny fixture.

Commit:

```text
feat(validation): add yearly backtesting
```

## Task F - Compare Recency Strategies

Use: `cavecrew-builder`.

Priority: P1

Own files:

- `src/aac_adoption/models/split.py`
- new or existing analysis script under `scripts/`
- tests only if needed: `tests/test_split.py`

Problem:

- Recency weights are fixed and wired, but no strategy comparison exists.

Acceptance:

- Compare full-history, recent 5-year, recent 3-year, and recency-weighted training on same final test period.
- Output table records strategy, train years, test years, subset, model, metrics.
- Test proves 2021 weight > 2013 weight remains true.

Commit:

```text
feat(validation): compare recency strategies
```

## Task G - Align Dashboard LOS Language

Use: `cavecrew-investigator`, then `cavecrew-builder`.

Priority: P1

Own files:

- `streamlit_app.py`
- `src/aac_adoption/dashboard/`
- `docs/METHODOLOGY.md`
- generated report templates only if needed

Problem:

- Roadmap still says reports/dashboard may conflate generic LOS with days to adoption.

Acceptance:

- Generic model text says `days to outcome` / `length of stay`.
- Adopted-only text says `days to adoption`.
- Sensitivity demo avoids exact-day certainty unless uncertainty bucket shown.
- Tests or grep check forbid bad phrases in dashboard/report source.

Investigator query:

```text
Find "time to adoption", "adoption speed", "predicted wait", "days to adoption", "days to outcome", "length of stay" in dashboard/report docs. Return path:line only.
```

Commit:

```text
fix(copy): separate LOS and adoption timing
```

## Task H - Enforce Subgroup Reliability Rules

Use: `cavecrew-builder`.

Priority: P1

Own files:

- `src/aac_adoption/reporting/evidence_pack.py`
- `src/aac_adoption/analysis/calibration_summary.py`
- tests only if needed: `tests/test_evidence_pack.py`

Problem:

- Subgroup reliability exists, but minimum sample-size interpretation rules are not fully accepted.

Acceptance:

- Subgroup outputs include sample-size flag, class-variety flag, and interpretation status.
- PR-AUC shown only where enough positive/negative cases exist.
- Generated summaries avoid interpreting cohorts below threshold.
- Tests cover small cohort and single-class cohort.

Commit:

```text
fix(reliability): guard subgroup interpretation
```

## Task I - Cluster-Aware Confidence Intervals

Use: `cavecrew-builder`.

Priority: P2

Own files:

- `src/aac_adoption/reporting/evidence_pack.py`
- `src/aac_adoption/models/evaluate.py`
- tests only if needed: `tests/test_evidence_pack.py`

Problem:

- Bootstrap appears row-level; same animal can appear multiple times.

Acceptance:

- Bootstrap by `animal_id` when available.
- Output records cluster count and bootstrap unit.
- If `animal_id` missing, fallback row-level with explicit limitation note.
- Test verifies repeated animal rows are resampled together.

Commit:

```text
fix(stats): bootstrap by animal
```

## Task J - Clean Roadmap Status Drift

Use: `cavecrew-builder`.

Priority: P2

Own files:

- `docs/ROADMAP.md`
- `docs/internal/plan_0606.md`
- `docs/internal/ml_hardening_done_reference.md`

Problem:

- Roadmap has contradictions: calibration marked DONE and RISK; horizon classifiers marked DONE while censoring summary remains open; old task doc claimed all tasks complete.

Acceptance:

- DONE only when code, test, and reproducible artifact path exist.
- PARTIAL when code exists but artifacts/docs/status incomplete.
- RISK only for unresolved methodological hazards.
- Reference file lists evidence for completed items.

Commit:

```text
docs(ml): reconcile hardening status
```

## Suggested Cavecrew Order

Round 1:

- Task A: calibration verification.
- Task B: LOS target leakage.
- Task C: censoring summary.

Round 2:

- Task D: re-intake ambiguity.
- Task E: yearly backtesting.
- Task G: dashboard LOS language.

Round 3:

- Task F: recency strategy comparison.
- Task H: subgroup reliability.
- Task I: cluster-aware CI.
- Task J: roadmap drift cleanup.

Final gate:

```powershell
python -m pytest -q
python scripts/calibrate_classifiers.py --help
python scripts/generate_artifact_manifest.py
```
