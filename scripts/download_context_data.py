"""CLI for downloading external AAC context datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aac_adoption.data.context_data import (  # noqa: E402
    DEFAULT_AUSTIN_WEATHER_STATION,
    download_austin_311_animal_requests,
    download_weather_daily,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Austin context data for AAC adoption features")
    parser.add_argument("--output-dir", default="data/raw/context")
    parser.add_argument("--start-date", default="2013-10-01")
    parser.add_argument("--end-date", default="2025-05-05")
    parser.add_argument("--weather-station", default=DEFAULT_AUSTIN_WEATHER_STATION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    weather_path = download_weather_daily(
        output_dir / "austin_weather_daily.csv",
        start_date=args.start_date,
        end_date=args.end_date,
        station=args.weather_station,
    )
    requests_path = download_austin_311_animal_requests(
        output_dir / "austin_311_animal_requests.csv",
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(f"Wrote weather context to {weather_path}")
    print(f"Wrote Austin 311 animal context to {requests_path}")


if __name__ == "__main__":
    main()
