import pandas as pd

from beijing_aqi.aqi import add_aqi_columns, aqi_category, calculate_index


def test_calculate_index_matches_pm25_breakpoint():
    assert round(calculate_index(12, "PM2.5"), 2) == 50.00
    assert round(calculate_index(35.4, "PM2.5"), 2) == 100.00


def test_aqi_category_labels():
    assert aqi_category(50) == "Good"
    assert aqi_category(100) == "Moderate"
    assert aqi_category(151) == "Unhealthy"
    assert aqi_category(301) == "Hazardous"
    assert aqi_category(float("nan")) is None


def test_add_aqi_columns_uses_max_component_index():
    frame = pd.DataFrame(
        {
            "PM2.5_24hr_avg": [12],
            "PM10_24hr_avg": [154],
            "SO2_24hr_avg": [0],
            "NO2_24hr_avg": [0],
            "CO_8hr_max": [0],
            "O3_8hr_max": [0],
        }
    )

    result = add_aqi_columns(frame)

    assert result.loc[0, "AQI_calculated"] == 100
    assert result.loc[0, "AQI_category"] == "Moderate"
