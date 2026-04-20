"""Download and extract the official UCI Beijing air-quality dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile


UCI_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/"
    "PRSA2017_Data_20130301-20170228.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Beijing AQI dataset.")
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where the dataset zip and extracted CSV files are stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even if the zip file already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "PRSA2017_Data_20130301-20170228.zip"
    extract_dir = output_dir / "PRSA_Data_20130301-20170228"

    if args.force or not zip_path.exists():
        print(f"Downloading {UCI_DATA_URL}")
        urlretrieve(UCI_DATA_URL, zip_path)
    else:
        print(f"Using existing {zip_path}")

    with ZipFile(zip_path) as archive:
        archive.extractall(output_dir)

    csv_count = len(list(extract_dir.glob("*.csv")))
    print(f"Extracted {csv_count} CSV files to {extract_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
