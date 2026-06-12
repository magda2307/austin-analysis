# Phase 5 Acceptance Architecture Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development`. Every behavior change follows
> red-green-refactor.

**Goal:** Make Phase 5 acceptance prove one complete, eligible `thesis-full` run
and independently verify every required artifact's existence, hash, freshness,
and receipt lineage.

**Architecture:** Separate producer-run eligibility, final manifest generation,
and verification-only acceptance. The pipeline writes and validates receipts but
never generates the final manifest. Manifest generation requires an explicit run
ID and fails atomically on invalid lineage. Acceptance reads existing artifacts
and receipts without mutating them.

**Tech Stack:** Python 3.12, pytest, pandas, PowerShell.

---

## Fixed Contracts

- Pipeline step 17 validates receipts; step 18 runs tests.
- Partial, quick, skipped-producer, and continue-on-error runs are never
  acceptance-eligible.
- `generate_artifact_manifest.py` requires `--run-id`.
- Manifest generation is atomic and fails on missing, empty, duplicate, stale,
  hash-invalid, cross-run, wrong-profile, failed-receipt, or missing-source
  artifacts.
- Required registry includes core datasets, model selection outputs, selected
  model paths and sidecars, report evidence, and final-facing documentation.
- `validate_final_acceptance.ps1 -Long` is verification-only.
- `-Long -SkipPytest` is invalid.
- Fixture acceptance consumes a caller-created fixture root read-only.

## Task 1: Strict Artifact Manifest And Acceptance Validator

**Files:**

- Create: `src/aac_adoption/acceptance.py`
- Modify: `scripts/generate_artifact_manifest.py`
- Rewrite: `tests/test_artifact_manifest.py`

**TDD sequence:**

1. Add failing fixtures for valid single-run lineage.
2. Add one failing test each for missing, empty, duplicate, missing-source,
   hash mismatch, stale manifest, missing receipt, failed receipt, cross-run,
   mixed source SHA, wrong profile, and receipt/manifest hash mismatch.
3. Add failing CLI tests proving `--run-id` is required, unknown runs fail
   without output mutation, and only the requested run is used.
4. Implement reusable read-only validation in `acceptance.py`.
5. Make manifest generation validate all inputs before atomic CSV/Markdown
   replacement.
6. Preserve adopted-only timing wording tests.

**Gate:**

```powershell
python -m pytest tests/test_artifact_manifest.py -q
python scripts/generate_artifact_manifest.py --help
```

## Task 2: Receipt Eligibility And Pipeline Ordering

**Files:**

- Create: `src/aac_adoption/run_receipt_validation.py`
- Create: `scripts/validate_run_receipts.py`
- Modify: `scripts/run_full_pipeline.py`
- Modify: `scripts/manage_run_context.py`
- Modify: `tests/test_pipeline_runner.py`
- Create: `tests/test_run_receipt_validation.py`

**TDD sequence:**

1. Add failing tests for step 17 receipt validation and step 18 pytest.
2. Add failing tests proving quick skips step 18 only.
3. Add failing tests for top-level receipt fields `completeness`,
   `eligibility_profile`, `profile`, and full source SHA.
4. Add failing strict validation tests for partial, skip-heavy,
   continue-on-error, failed, mixed-run, mixed-SHA, and incomplete receipts.
5. Implement shared validator and thin CLI.
6. Label only complete no-skip producer runs `thesis-full` and eligible.
7. Keep default fail-fast and keep final manifest outside pipeline.

**Gate:**

```powershell
python -m pytest tests/test_pipeline_runner.py tests/test_run_receipt_validation.py -q
python scripts/run_full_pipeline.py --help
python scripts/validate_run_receipts.py --help
```

## Task 3: Verification-Only PowerShell Acceptance

**Files:**

- Modify: `scripts/validate_final_acceptance.ps1`
- Create: `tests/test_acceptance_script.py`

**TDD sequence:**

1. Add a failing orchestration test for `-Long -SkipPytest`.
2. Add failing tests proving `-Long` invokes full pytest once, sets
   `AAC_ACCEPTANCE=1`, validates existing receipts, and runs no producers.
3. Add failing tests proving short smoke paths stay under one temporary root
   and cleanup occurs on success and failure.
4. Add an injectable Python executable for deterministic orchestration tests.
5. Implement early invalid-flag rejection and `try/finally` cleanup.

**Gate:**

```powershell
python -m pytest tests/test_acceptance_script.py -q
powershell -ExecutionPolicy Bypass -File scripts/validate_final_acceptance.ps1 -Long -SkipPytest
```

The PowerShell command must exit nonzero before running tests.

## Task 4: Adversarial Integration Verification

Run:

```powershell
python -m pytest tests/test_pipeline_runner.py tests/test_run_receipt_validation.py tests/test_artifact_manifest.py tests/test_acceptance_script.py tests/test_recency_comparison.py -q
python scripts/run_full_pipeline.py --help
python scripts/generate_artifact_manifest.py --help
python scripts/validate_run_receipts.py --help
python scripts/compare_recency.py --help
```

Then run fixture acceptance with valid and intentionally corrupt temporary
roots. Canonical acceptance may remain failed until manual clean regeneration;
that is correct behavior.

## Task 5: Closeout Evidence

Only after validators approve:

- update `docs/closeout/deep_testing_report_phase5.md`;
- update `docs/closeout/phase5.md`;
- update `.agents/CLOSEOUT.md`;
- record exact commands, exit codes, pass/fail counts, and remaining manual
  regeneration blocker.

Phase 5 may be marked architecture PASS while canonical artifacts remain
regeneration-blocked only if fixture tests prove every required failure mode and
canonical acceptance fails honestly on stale/missing artifacts.
