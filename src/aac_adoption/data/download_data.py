"""Download AAC raw CSV files from Austin Open Data."""

from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


SOCRATA_BASE = "https://data.austintexas.gov/api/views"


@dataclass(frozen=True)
class AacDatasetSource:
    """Socrata dataset metadata needed for raw CSV download."""

    name: str
    intakes_id: str
    outcomes_id: str
    description: str

    @property
    def intakes_url(self) -> str:
        return f"{SOCRATA_BASE}/{self.intakes_id}/rows.csv?accessType=DOWNLOAD"

    @property
    def outcomes_url(self) -> str:
        return f"{SOCRATA_BASE}/{self.outcomes_id}/rows.csv?accessType=DOWNLOAD"


DATASET_SOURCES = {
    "historical": AacDatasetSource(
        name="historical",
        intakes_id="wter-evkm",
        outcomes_id="9t4d-g238",
        description="Austin Animal Center records from 2013-10-01 to 2025-05-05",
    ),
    "current": AacDatasetSource(
        name="current",
        intakes_id="pyqf-r2dc",
        outcomes_id="gsvs-ypi7",
        description="Austin Animal Center ShelterBuddy records from 2025-05-05 onward",
    ),
}


def download_file(url: str, output_path: str | Path, overwrite: bool = False) -> Path:
    """Download URL to output path with simple atomic replacement."""
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass overwrite=True to replace it")

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_suffix(output.suffix + ".tmp")

    request = Request(url, headers={"User-Agent": "aac-adoption-thesis/0.1"})
    try:
        with urlopen(request, timeout=120) as response, temp_output.open("wb") as file:
            file.write(response.read())
    except URLError as exc:
        if temp_output.exists():
            temp_output.unlink()
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    temp_output.replace(output)
    return output


def download_aac_raw_data(
    output_dir: str | Path = "data/raw",
    source_name: str = "historical",
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Download intake and outcome CSV files for a configured AAC source."""
    if source_name not in DATASET_SOURCES:
        valid = ", ".join(sorted(DATASET_SOURCES))
        raise ValueError(f"Unknown source '{source_name}'. Valid sources: {valid}")

    source = DATASET_SOURCES[source_name]
    output = Path(output_dir)
    intakes_path = output / "intakes.csv"
    outcomes_path = output / "outcomes.csv"

    download_file(source.intakes_url, intakes_path, overwrite=overwrite)
    download_file(source.outcomes_url, outcomes_path, overwrite=overwrite)
    return intakes_path, outcomes_path

