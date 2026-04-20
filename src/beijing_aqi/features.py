"""Model feature definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd


REGRESSION_FEATURES = [
    "TEMP",
    "PRES",
    "PM2.5",
    "PM10",
    "O3_ppm",
    "CO_ppm",
    "SO2_ppb",
    "NO2_ppb",
]

CLASSIFICATION_FEATURES = [
    "TEMP",
    "PRES",
    "PM2.5",
    "O3_ppm",
    "CO_ppm",
    "SO2_ppb",
    "NO2_ppb",
]


def regression_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    dataset = frame[REGRESSION_FEATURES + ["AQI_calculated"]]
    dataset = dataset.replace([np.inf, -np.inf], pd.NA).dropna()
    return dataset[REGRESSION_FEATURES], dataset["AQI_calculated"]


def classification_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    dataset = frame[CLASSIFICATION_FEATURES + ["AQI_category"]]
    dataset = dataset.replace([np.inf, -np.inf], pd.NA).dropna()
    return dataset[CLASSIFICATION_FEATURES], dataset["AQI_category"]
