"""Validate the manually maintained documents included in final acceptance."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_DOCUMENTS = [
    ROOT / "README.md",
    ROOT / "docs" / "METHODOLOGY.md",
    ROOT / "docs" / "RESULTS.md",
    ROOT / "docs" / "target_definitions.md",
]


def main() -> None:
    missing = [path for path in FINAL_DOCUMENTS if not path.is_file()]
    empty = [path for path in FINAL_DOCUMENTS if path.is_file() and path.stat().st_size == 0]
    if missing or empty:
        problems = [f"missing: {path}" for path in missing]
        problems.extend(f"empty: {path}" for path in empty)
        raise SystemExit("\n".join(problems))
    for path in FINAL_DOCUMENTS:
        print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
