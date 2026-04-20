"""Data loading and preparation for Beijing air-quality records."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from beijing_aqi.aqi import add_aqi_columns


RAW_COLUMNS = [
    "year",
    "month",
    "day",
    "hour",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "wd",
    "WSPM",
    "station",
    "PM2.5",
    "PM10",
]

MEDIAN_FILL_COLUMNS = [
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
    "PM10",
    "PM2.5",
    "SO2",
    "NO2",
    "O3",
]

ZERO_FILL_COLUMNS = ["SO2", "NO2", "CO", "O3", "PM2.5", "PM10", "wd"]

MOLECULAR_WEIGHTS = {
    "O3": 47.998,
    "CO": 28.01,
    "SO2": 64.065,
    "NO2": 46.006,
}


def load_raw_data(data_dir: str | Path) -> pd.DataFrame:
    """Load and concatenate all station CSV files from a directory."""
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_path}. Download and unzip the UCI data first."
        )

    frames = [pd.read_csv(file_path, low_memory=False) for file_path in csv_files]
    raw = pd.concat(frames, axis=0, ignore_index=True)

    missing_columns = sorted(set(RAW_COLUMNS) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")

    return raw[RAW_COLUMNS].copy()


def clean_air_quality_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw station readings and derive time columns."""
    cleaned = raw.copy()
    cleaned["Date_time"] = pd.to_datetime(cleaned[["year", "month", "day", "hour"]])
    cleaned["Date"] = pd.to_datetime(cleaned[["year", "month", "day"]])
    cleaned = cleaned.drop_duplicates()

    grouped_medians = cleaned.groupby(["Date", "station"])[MEDIAN_FILL_COLUMNS].transform(
        "median"
    )
    for column in MEDIAN_FILL_COLUMNS:
        cleaned[column] = cleaned[column].fillna(grouped_medians[column])
        fallback = cleaned[column].median()
        if pd.isna(fallback):
            fallback = 0
        cleaned[column] = cleaned[column].fillna(fallback)

    pressure_fallback = cleaned.loc[cleaned["PRES"] > 0, "PRES"].median()
    if not pd.isna(pressure_fallback):
        cleaned.loc[cleaned["PRES"] <= 0, "PRES"] = pressure_fallback

    for column in ZERO_FILL_COLUMNS:
        cleaned[column] = cleaned[column].fillna(0)

    return cleaned


def add_pollutant_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add gas unit conversions and rolling-window pollutant features."""
    result = frame.copy()

    result["O3_ppm"] = _micrograms_to_ppm(result, "O3")
    result["CO_ppm"] = _micrograms_to_ppm(result, "CO")
    result["SO2_ppb"] = _micrograms_to_ppb(result, "SO2")
    result["NO2_ppb"] = _micrograms_to_ppb(result, "NO2")

    result["TEMP"] = result["TEMP"] + 273

    result["PM10_24hr_avg"] = _rolling_by_station(result, "PM10", window=24, method="mean")
    result["PM2.5_24hr_avg"] = _rolling_by_station(result, "PM2.5", window=24, method="mean")
    result["SO2_24hr_avg"] = _rolling_by_station(result, "SO2_ppb", window=24, method="mean")
    result["NO2_24hr_avg"] = _rolling_by_station(result, "NO2_ppb", window=24, method="mean")
    result["CO_8hr_max"] = _rolling_by_station(result, "CO_ppm", window=8, method="max")
    result["O3_8hr_max"] = _rolling_by_station(result, "O3_ppm", window=8, method="max")

    return result


def build_feature_frame(data_dir: str | Path) -> pd.DataFrame:
    """Load raw CSVs and return a model-ready AQI feature frame."""
    raw = load_raw_data(data_dir)
    cleaned = clean_air_quality_data(raw)
    pollutant_features = add_pollutant_features(cleaned)
    return add_aqi_columns(pollutant_features)


def _micrograms_to_ppm(frame: pd.DataFrame, pollutant: str) -> pd.Series:
    return (
        (frame[pollutant] / 1000)
        * (22.4 / MOLECULAR_WEIGHTS[pollutant])
        * ((273 + frame["TEMP"]) / 273)
        * (1013 / frame["PRES"])
    )


def _micrograms_to_ppb(frame: pd.DataFrame, pollutant: str) -> pd.Series:
    return (
        frame[pollutant]
        * (22.4 / MOLECULAR_WEIGHTS[pollutant])
        * ((273 + frame["TEMP"]) / 273)
        * (1013 / frame["PRES"])
    )


def _rolling_by_station(
    frame: pd.DataFrame, column: str, window: int, method: str
) -> pd.Series:
    grouped = frame.groupby("station", group_keys=False)[column]

    if method == "mean":
        return grouped.transform(lambda values: values.rolling(window=window).mean())
    if method == "max":
        return grouped.transform(lambda values: values.rolling(window=window).max())

    raise ValueError(f"Unsupported rolling method: {method}")
