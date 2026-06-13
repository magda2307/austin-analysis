# Agent Notes for the PJATK Thesis LaTeX Project

Scope: this file applies to every file in `szablon-pracy-dyplomowej-pjatk-master/`.

## Project Context

- The main compile entry point is `glowny.tex`; chapter files are included with `\input{...}` and should not contain their own preamble.
- The thesis is written in Polish. Preserve a formal academic tone and avoid switching section prose to English except for established technical terms already present in the text.
- The active results chapter is `rozdzial5.tex`, titled `Wyniki badań i interpretacja modeli`; it summarizes model evaluation, hypothesis verification, SHAP/XAI interpretation, calibration, and deployment-facing implications.
- The empirical framing in chapter 5 is a chronological split: 2013--2021 training, 2022--2023 validation/threshold selection, and 2024--2025 one-time test.
- The chapter consistently separates two modeling tasks: adoption classification and regression of days to any matched outcome, not only adoption.

## Editing Rules

- Keep edits minimal and source-backed; do not change reported metrics, dates, hypothesis statuses, or model rankings unless the underlying analysis artifacts are also updated.
- Use Polish decimal commas in prose and tables, for example `0,840` and `18,55`, while preserving code-like identifiers in `\texttt{...}`.
- Preserve existing label conventions such as `chap:...`, `sec:...`, `fig:...`, and `tab:...`; update every `\ref{...}` if a label changes.
- Prefer relative paths in LaTeX, especially for figures and inputs. Do not add absolute paths.
- Do not write generated preview images or other temporary render artifacts into this directory; use `/tmp` locations for previews.
- Avoid adding new packages in `glowny.tex` unless a change cannot be expressed with the packages already loaded.

## Chapter 5 Conventions

- Use `\ChapterFiveFigure{path}{caption}{label}` for chapter 5 figures so missing figure files render as explicit placeholders instead of breaking compilation.
- Keep the key classification conclusion intact unless revalidated: HistGradientBoosting is reported as the best test classifier, with ROC-AUC around `0,840` on the combined test set.
- Keep the key regression conclusion intact unless revalidated: CatBoost is reported as the best test regressor, with MAE around `18,55` days on the combined test set.
- Preserve the methodological caveat that `days_to_adoption` is descriptive only and that the regression target is days to any matched outcome.
- Treat SHAP/XAI descriptions as interpretive support, not causal proof; maintain wording that distinguishes statistical associations from causal claims.
- Maintain caution around small test subgroups, calibration, and threshold transfer from validation to test.

## Compilation Notes

- Compile from this directory with `pdflatex glowny.tex`, then `bibtex glowny`, then two more `pdflatex glowny.tex` runs when bibliography or references change.
- Generated LaTeX auxiliary files such as `.aux`, `.log`, `.toc`, `.lof`, `.out`, `.fls`, `.fdb_latexmk`, `.synctex`, and PDFs are build artifacts; do not edit them by hand.