# Phase 3 Knowledge

*Log testing gotchas, shortcuts, and context discoveries here.*

### Testing Gotchas
- **Pandas `str.contains` with NaN**: When adding skipped rows with `None` values to output DataFrames (such as the backtesting evaluation results), pandas `.str.contains(...)` will return `NaN` for those rows and raise a `ValueError: Cannot mask with non-boolean array containing NA / NaN values`. Always use `.str.contains(..., na=False)` when filtering to safely ignore `None` values.

### Context Discoveries
- **Report Generation Filtering**: Model comparison CSVs may contain multiple splits (e.g. `selection` and `test` from backtesting). Always filter the pandas DataFrame explicitly for `metric_split == "selection"` before calling functions like `idxmax()` to ensure you're picking the model that performed best on the validation set, not the test set.
- **CLI Argument Passing**: Always ensure arguments parsed via `argparse` in `scripts/` (e.g. `data_path`, `diagnostics_dir`) are explicitly passed to the underlying runner functions. Do not rely on canonical fallback logic if an explicit directory parameter is available.
