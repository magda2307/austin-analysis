# Compact Repository Context

## Choose One Packet

| Task | Minimal context |
|---|---|
| File/bug already known | Target file + matching test row in `COMMANDS.md` |
| Need owner/call chain | `REPO_MAP.md`, then named source files |
| Closeout blocker | `CLOSEOUT.md`, then exact task heading in detailed checklist |
| Target or leakage semantics | Named section in `docs/target_definitions.md` |
| Thesis claim/wording | Named section in `docs/METHODOLOGY.md` |
| Test or regeneration | `COMMANDS.md` |
| Generated output mismatch | Producer from `REPO_MAP.md`; do not patch artifact |

`README.md`, `docs/ROADMAP.md`, and `docs/RESULTS.md` currently contain stale or
contradictory sections. They are closeout outputs to reconcile, not reliable
implementation specifications.

Historical Markdown archives and one-off agent plans were removed from
`docs/old`, `docs/internal`, and the repository root. Use git history when old
context is truly needed; do not recreate archival copies in the worktree.

## Core Flow

```text
raw AAC CSVs
  -> clean and match episodes
  -> build modeling_dataset.csv + metadata
  -> train chronological classification/regression models
  -> compare/select/calibrate/diagnose
  -> generate reports and manifest
  -> Streamlit reads artifacts
```

## Fast Orientation

- Package: `src/aac_adoption/`
- CLI entrypoints: `scripts/`
- Tests: `tests/`
- Dashboard entrypoint: `streamlit_app.py`
- Generated/local data: `data/processed/`
- Generated models: `models/`
- Generated evidence: `reports/`
- Canonical orchestrator: `scripts/run_full_pipeline.py`
- Canonical acceptance helper: `scripts/validate_final_acceptance.ps1`

## Truth Precedence

For current behavior:

```text
source + direct tests > generated artifacts > current docs > historical notes
```

For intended thesis semantics, `docs/target_definitions.md` wins. When source
violates it, record/fix a bug rather than rewriting the definition around code.

## Low-Context Mechanics

```powershell
rg -n "^(#|##|###) " <large-markdown-file>
rg -n "symbol|column|artifact" src tests scripts
Get-Content <file> | Select-Object -Skip <start> -First <count>
```

Do not load all of root `README.md` (about 58 KB) or `agentsbatch*/` (historical
multi-agent logs). The former generated `THESIS_CONTEXT_FOR_LLM.md` megadocument
and its compiler were removed; use heading search and bounded source reads.

Before edits: `git status --short`, target source, direct tests. Read target
definitions only for target/leakage/wording work; read closeout only for closeout
work. Run narrow tests before broad tests.

Before deleting a suspected stale file, prove no live inbound references:

```powershell
git ls-files -- <path>
rg -n --fixed-strings "<file-name>" . --glob "!agentsbatch*/**"
git log --oneline --all -- <path>
```
