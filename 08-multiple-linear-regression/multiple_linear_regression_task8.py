from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT / "task8_weather_pm10_hourly_master.csv"
TEST_PREDICTIONS_CSV = ROOT / "linear_regression_test_predictions.csv"
JUNE_FORECAST_CSV = ROOT / "june_forecast_linear_regression.csv"
METRICS_CSV = ROOT / "linear_regression_metrics.csv"

TEST_DAYS = 15
FORECAST_START = pd.Timestamp("2026-06-01 00:00:00")
FORECAST_END = pd.Timestamp("2026-06-30 23:00:00")

TARGETS = {
    "temperature": {
        "column": "air_temperature_c",
        "label": "Temperature",
        "actual": "actual_temperature_c",
        "predicted": "predicted_temperature_c",
        "error": "temperature_error_c",
        "absolute_error": "temperature_absolute_error_c",
        "relative_error": "temperature_relative_error_percent",
        "unit": "C",
    },
    "pm10": {
        "column": "pm10_outdoors_ug_m3",
        "label": "PM10",
        "actual": "actual_pm10_ug_m3",
        "predicted": "predicted_pm10_ug_m3",
        "error": "pm10_error_ug_m3",
        "absolute_error": "pm10_absolute_error_ug_m3",
        "relative_error": "pm10_relative_error_percent",
        "unit": "ug/m3",
    },
}

TARGET_COLUMNS = ["air_temperature_c", "pm10_outdoors_ug_m3"]
FORECAST_FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "dayofweek",
    "month",
    "dayofyear_sin",
    "dayofyear_cos",
    "temperature_lag_1h",
    "temperature_lag_2h",
    "temperature_lag_3h",
    "temperature_lag_6h",
    "temperature_lag_12h",
    "temperature_lag_24h",
    "temperature_lag_48h",
    "temperature_rolling_3h",
    "temperature_rolling_6h",
    "temperature_rolling_24h",
    "temperature_rolling_168h",
    "pm10_lag_1h",
    "pm10_lag_2h",
    "pm10_lag_3h",
    "pm10_lag_6h",
    "pm10_lag_12h",
    "pm10_lag_24h",
    "pm10_lag_48h",
    "pm10_rolling_3h",
    "pm10_rolling_6h",
    "pm10_rolling_24h",
    "pm10_rolling_168h",
]


def load_source_data() -> pd.DataFrame:
    data = pd.read_csv(SOURCE_CSV, parse_dates=["timestamp"])
    data = data.sort_values("timestamp").set_index("timestamp")
    return data


def add_model_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    idx = data.index

    data["hour"] = idx.hour
    data["dayofweek"] = idx.dayofweek
    data["month"] = idx.month
    data["dayofyear"] = idx.dayofyear

    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["dayofyear_sin"] = np.sin(2 * np.pi * data["dayofyear"] / 366)
    data["dayofyear_cos"] = np.cos(2 * np.pi * data["dayofyear"] / 366)

    wind_rad = np.deg2rad(data["wind_direction_deg"])
    data["wind_direction_sin"] = np.sin(wind_rad)
    data["wind_direction_cos"] = np.cos(wind_rad)

    lag_columns = {
        "air_temperature_c": "temperature",
        "pm10_outdoors_ug_m3": "pm10",
        "relative_humidity_percent": "humidity",
        "wind_speed_m_s": "wind_speed",
        "precipitation_mm": "precipitation",
        "pyranometer_w_m2": "pyranometer",
    }
    for source_col, prefix in lag_columns.items():
        data[f"{prefix}_lag_1h"] = data[source_col].shift(1)
        data[f"{prefix}_lag_24h"] = data[source_col].shift(24)
        data[f"{prefix}_rolling_3h"] = data[source_col].shift(1).rolling(3, min_periods=2).mean()
        data[f"{prefix}_rolling_24h"] = data[source_col].shift(1).rolling(24, min_periods=8).mean()

    return data


def feature_columns_for_target(target_column: str) -> list[str]:
    feature_columns = [
        "air_temperature_c",
        "pm10_outdoors_ug_m3",
        "precipitation_mm",
        "pyranometer_w_m2",
        "relative_humidity_percent",
        "wind_direction_deg",
        "wind_speed_m_s",
        "wind_direction_sin",
        "wind_direction_cos",
        "hour_sin",
        "hour_cos",
        "dayofyear_sin",
        "dayofyear_cos",
        "month",
        "dayofweek",
        "temperature_lag_1h",
        "temperature_lag_24h",
        "temperature_rolling_3h",
        "temperature_rolling_24h",
        "pm10_lag_1h",
        "pm10_lag_24h",
        "pm10_rolling_3h",
        "pm10_rolling_24h",
        "humidity_lag_1h",
        "humidity_lag_24h",
        "humidity_rolling_3h",
        "humidity_rolling_24h",
        "wind_speed_lag_1h",
        "wind_speed_lag_24h",
        "wind_speed_rolling_3h",
        "wind_speed_rolling_24h",
        "precipitation_lag_1h",
        "precipitation_lag_24h",
        "precipitation_rolling_3h",
        "precipitation_rolling_24h",
        "pyranometer_lag_1h",
        "pyranometer_lag_24h",
        "pyranometer_rolling_3h",
        "pyranometer_rolling_24h",
    ]
    return [col for col in feature_columns if col != target_column]


def split_latest_15_days(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    latest_timestamp = data.index.max()
    test_start = latest_timestamp - pd.Timedelta(days=TEST_DAYS)
    train = data[data.index < test_start]
    test = data[data.index >= test_start]
    return train, test, test_start


def relative_error_percent(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    denominator = np.abs(actual)
    result = np.full_like(predicted, np.nan, dtype=float)
    valid = denominator > 1e-12
    result[valid] = (np.abs(predicted[valid] - actual[valid]) / denominator[valid]) * 100
    return result


def train_and_predict_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_key: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    target = TARGETS[target_key]
    target_column = target["column"]
    feature_columns = feature_columns_for_target(target_column)
    required_columns = feature_columns + [target_column]

    train_ready = train.dropna(subset=required_columns)
    test_ready = test.dropna(subset=required_columns)

    X_train = train_ready[feature_columns]
    y_train = train_ready[target_column]
    X_test = test_ready[feature_columns]
    y_test = test_ready[target_column]

    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    actual = y_test.to_numpy(dtype=float)
    error = predicted - actual
    absolute_error = np.abs(error)
    mse = mean_squared_error(actual, predicted)

    predictions = pd.DataFrame(
        {
            "timestamp": test_ready.index,
            target["actual"]: actual,
            target["predicted"]: predicted,
            target["error"]: error,
            target["absolute_error"]: absolute_error,
            target["relative_error"]: relative_error_percent(actual, predicted),
        }
    )

    metrics = {
        "model": "Multiple Linear Regression",
        "target": target["label"],
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_start": train_ready.index.min(),
        "train_end": train_ready.index.max(),
        "test_start": test_ready.index.min(),
        "test_end": test_ready.index.max(),
        "mae": mean_absolute_error(actual, predicted),
        "mse": mse,
        "rmse": mse**0.5,
        "r2": r2_score(actual, predicted),
        "mean_relative_error_percent": np.nanmean(relative_error_percent(actual, predicted)),
    }
    return predictions, metrics


def make_test_predictions(featured_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    train, test, test_start = split_latest_15_days(featured_data)
    temperature_predictions, temperature_metrics = train_and_predict_target(train, test, "temperature")
    pm10_predictions, pm10_metrics = train_and_predict_target(train, test, "pm10")

    predictions = temperature_predictions.merge(pm10_predictions, on="timestamp", how="inner")
    metrics = pd.DataFrame([temperature_metrics, pm10_metrics])
    split_info = {
        "dataset_rows": len(featured_data),
        "dataset_start": featured_data.index.min(),
        "dataset_end": featured_data.index.max(),
        "test_start": test_start,
        "train_rows": len(train),
        "test_rows": len(test),
    }
    return predictions, metrics, split_info


def add_forecast_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    idx = data.index

    data["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    data["dayofweek"] = idx.dayofweek
    data["month"] = idx.month
    data["dayofyear_sin"] = np.sin(2 * np.pi * idx.dayofyear / 366)
    data["dayofyear_cos"] = np.cos(2 * np.pi * idx.dayofyear / 366)

    lag_map = {
        "air_temperature_c": "temperature",
        "pm10_outdoors_ug_m3": "pm10",
    }
    for column, prefix in lag_map.items():
        for lag in [1, 2, 3, 6, 12, 24, 48]:
            data[f"{prefix}_lag_{lag}h"] = data[column].shift(lag)
        data[f"{prefix}_rolling_3h"] = data[column].shift(1).rolling(3, min_periods=2).mean()
        data[f"{prefix}_rolling_6h"] = data[column].shift(1).rolling(6, min_periods=3).mean()
        data[f"{prefix}_rolling_24h"] = data[column].shift(1).rolling(24, min_periods=8).mean()
        data[f"{prefix}_rolling_168h"] = data[column].shift(1).rolling(168, min_periods=48).mean()

    return data


def train_forecast_models(data: pd.DataFrame) -> dict[str, LinearRegression]:
    featured = add_forecast_features(data)
    ready = featured.dropna(subset=FORECAST_FEATURE_COLUMNS + TARGET_COLUMNS)
    X_train = ready[FORECAST_FEATURE_COLUMNS]

    models = {}
    for target_column in TARGET_COLUMNS:
        model = LinearRegression()
        model.fit(X_train, ready[target_column])
        models[target_column] = model
    return models


def forecast_june(data: pd.DataFrame, models: dict[str, LinearRegression]) -> pd.DataFrame:
    history = data[TARGET_COLUMNS].copy()
    forecast_index = pd.date_range(
        start=history.index.max() + pd.Timedelta(hours=1),
        end=FORECAST_END,
        freq="h",
    )

    rows = []
    for timestamp in forecast_index:
        history.loc[timestamp, TARGET_COLUMNS] = np.nan
        featured = add_forecast_features(history)
        X_next = featured.loc[[timestamp], FORECAST_FEATURE_COLUMNS]

        predicted_temperature = float(models["air_temperature_c"].predict(X_next)[0])
        predicted_pm10 = max(float(models["pm10_outdoors_ug_m3"].predict(X_next)[0]), 0.0)

        history.loc[timestamp, "air_temperature_c"] = predicted_temperature
        history.loc[timestamp, "pm10_outdoors_ug_m3"] = predicted_pm10

        if FORECAST_START <= timestamp <= FORECAST_END:
            rows.append(
                {
                    "timestamp": timestamp,
                    "predicted_temperature_c": predicted_temperature,
                    "predicted_pm10_ug_m3": predicted_pm10,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    data = load_source_data()
    featured_data = add_model_features(data)

    predictions, metrics, split_info = make_test_predictions(featured_data)
    forecast_models = train_forecast_models(data[TARGET_COLUMNS].dropna())
    june_forecast = forecast_june(data[TARGET_COLUMNS].dropna(), forecast_models)

    predictions.to_csv(TEST_PREDICTIONS_CSV, index=False, float_format="%.4f")
    metrics.to_csv(METRICS_CSV, index=False, float_format="%.6f")
    june_forecast.to_csv(JUNE_FORECAST_CSV, index=False, float_format="%.4f")

    print("model=Multiple Linear Regression")
    print(f"source_csv={SOURCE_CSV.name}")
    print(f"source_start={split_info['dataset_start']}")
    print(f"source_end={split_info['dataset_end']}")
    print(f"test_start={split_info['test_start']}")
    print(f"test_rows={len(predictions)}")
    print(f"june_forecast_rows={len(june_forecast)}")
    print(f"test_predictions_csv={TEST_PREDICTIONS_CSV.name}")
    print(f"june_forecast_csv={JUNE_FORECAST_CSV.name}")
    print("")
    print(metrics[["target", "mae", "mse", "rmse", "r2", "mean_relative_error_percent"]].to_string(index=False))


if __name__ == "__main__":
    main()
