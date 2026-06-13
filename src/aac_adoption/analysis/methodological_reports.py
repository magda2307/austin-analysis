"""Generate concise methodological limitation summaries."""

from pathlib import Path


def create_methodological_reports(summary_dir: str | Path) -> None:
    summary = Path(summary_dir)
    summary.mkdir(parents=True, exist_ok=True)
    reports = {
        "external_validity_limitations.md": """# External validity limitations

Austin Animal Center operates in a specific No-Kill policy and service context.
Results are predictive and descriptive for AAC records from 2013-2025. They do
not automatically generalize to other shelters, cities, policies, or periods.
Replication on external shelter data is required.
""",
        "breed_color_justification.md": """# Breed and color representation

Breed and color values are grouped before modeling. Raw labels have high
granularity, inconsistent spelling, and sparse categories. Grouping reduces
sparsity and unstable estimates while preserving broad appearance information.
These variables are predictive descriptors, not causal explanations.
""",
        "descriptive_baseline_comparison.md": """# Descriptive baseline comparison

Model value is judged against a dummy baseline and simple descriptive rates.
Reported lift means improvement over that baseline on the chronological
evaluation period. Lift is predictive performance, not evidence of intervention
impact or causality.
""",
    }
    for filename, content in reports.items():
        (summary / filename).write_text(content.strip() + "\n", encoding="utf-8")
