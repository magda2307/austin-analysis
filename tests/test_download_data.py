from pathlib import Path

import pytest

from aac_adoption.data.download_data import DATASET_SOURCES, download_file


def test_historical_source_uses_expected_socrata_ids():
    source = DATASET_SOURCES["historical"]

    assert "wter-evkm" in source.intakes_url
    assert "9t4d-g238" in source.outcomes_url
    assert source.intakes_url.endswith("rows.csv?accessType=DOWNLOAD")


def test_download_file_refuses_overwrite(tmp_path: Path):
    output = tmp_path / "intakes.csv"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        download_file("https://example.com/file.csv", output, overwrite=False)

