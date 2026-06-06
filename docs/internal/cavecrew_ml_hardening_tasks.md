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





## Task F - Enforce Subgroup Reliability Rules

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

## Task G - Cluster-Aware Confidence Intervals

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

## Task H - Clean Roadmap Status Drift

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

- Task A: refresh censoring artifacts.
- Task B: re-intake ambiguity.
- Task C: yearly backtesting.

Round 2:

- Task D: recency strategy comparison.
- Task E: dashboard LOS language.
- Task F: subgroup reliability.

Round 3:

- Task G: cluster-aware CI.
- Task H: roadmap drift cleanup.

Final gate:

```powershell
python -m pytest -q
python scripts/calibrate_classifiers.py --help
python scripts/generate_artifact_manifest.py
```
