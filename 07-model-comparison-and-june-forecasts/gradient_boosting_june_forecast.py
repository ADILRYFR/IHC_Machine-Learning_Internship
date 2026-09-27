from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing modelling dependency. Install scikit-learn first:\n"
        "  python3 -m pip install scikit-learn\n"
        f"Original error: {exc}"
    ) from exc


SOURCE_CSV = Path(__file__).resolve().parent / "task7_weather_pm10_hourly_master.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "june_forecast_gradient_boosting.csv"

RANDOM_STATE = 42
FORECAST_START = pd.Timestamp("2026-06-01 00:00:00")
FORECAST_END = pd.Timestamp("2026-06-30 23:00:00")

TARGET_COLUMNS = ["air_temperature_c", "pm10_outdoors_ug_m3"]
FEATURE_COLUMNS = [
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


def train_models(data: pd.DataFrame) -> dict[str, HistGradientBoostingRegressor]:
    featured = add_forecast_features(data)
    ready = featured.dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS)

    models = {}
    X_train = ready[FEATURE_COLUMNS]
    for target in TARGET_COLUMNS:
        model = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.04,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, ready[target])
        models[target] = model
    return models


def forecast_june(data: pd.DataFrame, models: dict[str, HistGradientBoostingRegressor]) -> pd.DataFrame:
    history = data[TARGET_COLUMNS].copy()
    full_forecast_index = pd.date_range(
        start=history.index.max() + pd.Timedelta(hours=1),
        end=FORECAST_END,
        freq="h",
    )

    rows = []
    for timestamp in full_forecast_index:
        history.loc[timestamp, TARGET_COLUMNS] = np.nan
        featured = add_forecast_features(history)
        X_next = featured.loc[[timestamp], FEATURE_COLUMNS]

        predicted_temperature = float(models["air_temperature_c"].predict(X_next)[0])
        predicted_pm10 = float(models["pm10_outdoors_ug_m3"].predict(X_next)[0])

        history.loc[timestamp, "air_temperature_c"] = predicted_temperature
        history.loc[timestamp, "pm10_outdoors_ug_m3"] = max(predicted_pm10, 0.0)

        if FORECAST_START <= timestamp <= FORECAST_END:
            rows.append(
                {
                    "timestamp": timestamp,
                    "predicted_temperature_c": predicted_temperature,
                    "predicted_pm10_ug_m3": max(predicted_pm10, 0.0),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    data = pd.read_csv(SOURCE_CSV, parse_dates=["timestamp"])
    data = data.sort_values("timestamp").set_index("timestamp")
    data = data[TARGET_COLUMNS].dropna()

    models = train_models(data)
    forecast = forecast_june(data, models)
    forecast.to_csv(OUTPUT_CSV, index=False)

    print("model=HistGradientBoostingRegressor")
    print(f"source_csv={SOURCE_CSV.name}")
    print(f"output_csv={OUTPUT_CSV.name}")
    print(f"source_start={data.index.min()}")
    print(f"source_end={data.index.max()}")
    print(f"forecast_start={forecast['timestamp'].min()}")
    print(f"forecast_end={forecast['timestamp'].max()}")
    print(f"forecast_rows={len(forecast)}")


if __name__ == "__main__":
    main()
