"""AQI feature engineering utilities.

The breakpoint tables mirror the formulas used in the original notebook, but
the implementation is reusable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Breakpoint:
    concentration_low: float
    concentration_high: float
    index_low: int
    index_high: int


BREAKPOINTS: dict[str, tuple[Breakpoint, ...]] = {
    "PM2.5": (
        Breakpoint(0, 12, 0, 50),
        Breakpoint(12.1, 35.4, 51, 100),
        Breakpoint(35.5, 55.4, 101, 150),
        Breakpoint(55.5, 150.4, 151, 200),
        Breakpoint(150.5, 250.4, 201, 300),
        Breakpoint(250.5, 350.4, 301, 400),
        Breakpoint(350.5, 500.4, 401, 500),
    ),
    "PM10": (
        Breakpoint(0, 54, 0, 50),
        Breakpoint(55, 154, 51, 100),
        Breakpoint(155, 254, 101, 150),
        Breakpoint(255, 354, 151, 200),
        Breakpoint(355, 424, 201, 300),
        Breakpoint(425, 504, 301, 400),
        Breakpoint(504, 604, 401, 500),
    ),
    "SO2": (
        Breakpoint(0, 35, 0, 50),
        Breakpoint(36, 75, 51, 100),
        Breakpoint(76, 185, 101, 150),
        Breakpoint(186, 304, 151, 200),
        Breakpoint(305, 604, 201, 300),
        Breakpoint(605, 804, 301, 400),
        Breakpoint(805, 1004, 401, 500),
    ),
    "NO2": (
        Breakpoint(0, 53, 0, 50),
        Breakpoint(54, 100, 51, 100),
        Breakpoint(101, 360, 101, 150),
        Breakpoint(361, 649, 151, 200),
        Breakpoint(650, 1249, 201, 300),
        Breakpoint(1250, 1649, 301, 400),
        Breakpoint(1650, 2049, 401, 500),
    ),
    "CO": (
        Breakpoint(0, 4.4, 0, 50),
        Breakpoint(4.5, 9.4, 51, 100),
        Breakpoint(9.5, 12.4, 101, 150),
        Breakpoint(12.5, 15.4, 151, 200),
        Breakpoint(15.5, 30.4, 201, 300),
        Breakpoint(30.5, 40.4, 301, 400),
        Breakpoint(40.5, 50.4, 401, 500),
    ),
    "O3": (
        Breakpoint(0, 0.054, 0, 50),
        Breakpoint(0.055, 0.070, 51, 100),
        Breakpoint(0.071, 0.085, 101, 150),
        Breakpoint(0.086, 0.105, 151, 200),
        Breakpoint(0.106, 0.200, 201, 300),
        Breakpoint(0.201, 0.404, 301, 400),
        Breakpoint(0.405, 0.604, 401, 500),
    ),
}

AQI_COMPONENT_COLUMNS = {
    "PM2.5": "PM2.5_24hr_avg",
    "PM10": "PM10_24hr_avg",
    "SO2": "SO2_24hr_avg",
    "NO2": "NO2_24hr_avg",
    "CO": "CO_8hr_max",
    "O3": "O3_8hr_max",
}

INDEX_COLUMNS = [
    "PM2.5_Index",
    "PM10_Index",
    "SO2_Index",
    "NO2_Index",
    "CO_Index",
    "O3_Index",
]


def calculate_index(value: float, pollutant: str) -> float:
    """Calculate a pollutant-specific AQI component index."""
    if value is None:
        return 0

    numeric_value = float(value)
    if isnan(numeric_value) or numeric_value < 0:
        return 0

    for breakpoint in BREAKPOINTS[pollutant]:
        if numeric_value <= breakpoint.concentration_high:
            index_range = breakpoint.index_high - breakpoint.index_low
            concentration_range = (
                breakpoint.concentration_high - breakpoint.concentration_low
            )
            concentration_delta = numeric_value - breakpoint.concentration_low
            return breakpoint.index_low + (
                concentration_delta * index_range / concentration_range
            )

    return 0


def aqi_category(value: float) -> str | None:
    """Return the AQI category label for a final AQI value."""
    if value is None:
        return None

    numeric_value = float(value)
    if isnan(numeric_value):
        return None
    if numeric_value <= 50:
        return "Good"
    if numeric_value <= 100:
        return "Moderate"
    if numeric_value <= 150:
        return "Unhealthy for Sensitive Groups"
    if numeric_value <= 200:
        return "Unhealthy"
    if numeric_value <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def add_aqi_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add pollutant component indices, final AQI, and AQI category columns."""
    result = frame.copy()

    for pollutant, source_column in AQI_COMPONENT_COLUMNS.items():
        index_column = f"{pollutant}_Index"
        result[index_column] = result[source_column].apply(
            lambda value: calculate_index(value, pollutant)
        )

    result["Checks"] = (result[INDEX_COLUMNS] > 0).sum(axis=1)
    result["AQI_calculated"] = np.round(result[INDEX_COLUMNS].max(axis=1))
    result["AQI_category"] = result["AQI_calculated"].apply(aqi_category)
    return result[result["AQI_calculated"] != 0].copy()


def available_categories() -> Iterable[str]:
    return (
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    )
