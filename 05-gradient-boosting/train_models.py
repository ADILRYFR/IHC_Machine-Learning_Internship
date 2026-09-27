from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing modelling dependency. Install requirements first:\n"
        "  python3 -m pip install -r requirements.txt\n"
        f"Original error: {exc}"
    ) from exc


MASTER_CSV = Path(__file__).resolve().parent / "task5_weather_pm10_hourly_master.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

TEST_DAYS = 15
RANDOM_STATE = 42

TARGETS = {
    "temperature": {
        "column": "air_temperature_c",
        "prediction_file": "predictions_temperature.csv",
    },
    "pm10": {
        "column": "pm10_outdoors_ug_m3",
        "prediction_file": "predictions_pm10.csv",
    },
}


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


def train_and_predict(train: pd.DataFrame, test: pd.DataFrame, target_name: str, feature_columns: list[str]) -> dict[str, object]:
    target_column = TARGETS[target_name]["column"]
    prediction_file = TARGETS[target_name]["prediction_file"]

    required_columns = feature_columns + [target_column]
    train_ready = train.dropna(subset=required_columns)
    test_ready = test.dropna(subset=required_columns)

    X_train = train_ready[feature_columns]
    y_train = train_ready[target_column]
    X_test = test_ready[feature_columns]
    y_test = test_ready[target_column]

    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.04,
        l2_regularization=0.05,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    predictions = pd.DataFrame(
        {
            "timestamp": test_ready.index,
            "actual": y_test.to_numpy(),
            "predicted": predicted,
        }
    )
    predictions["error"] = predictions["predicted"] - predictions["actual"]
    predictions["absolute_error"] = predictions["error"].abs()
    predictions.to_csv(OUTPUT_DIR / prediction_file, index=False)

    mse = mean_squared_error(y_test, predicted)
    rmse = mse ** 0.5
    return {
        "target": target_name,
        "model": "HistGradientBoostingRegressor",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_start": train_ready.index.min(),
        "train_end": train_ready.index.max(),
        "test_start": test_ready.index.min(),
        "test_end": test_ready.index.max(),
        "mae": mean_absolute_error(y_test, predicted),
        "mse": mse,
        "rmse": rmse,
        "r2": r2_score(y_test, predicted),
    }


def save_combined_predictions() -> None:
    temperature = pd.read_csv(OUTPUT_DIR / TARGETS["temperature"]["prediction_file"], parse_dates=["timestamp"])
    pm10 = pd.read_csv(OUTPUT_DIR / TARGETS["pm10"]["prediction_file"], parse_dates=["timestamp"])

    combined = temperature.merge(pm10, on="timestamp", suffixes=("_temperature_c", "_pm10_ug_m3"))
    combined = combined.rename(
        columns={
            "actual_temperature_c": "actual_temperature_c",
            "predicted_temperature_c": "predicted_temperature_c",
            "error_temperature_c": "temperature_error_c",
            "absolute_error_temperature_c": "temperature_absolute_error_c",
            "actual_pm10_ug_m3": "actual_pm10_ug_m3",
            "predicted_pm10_ug_m3": "predicted_pm10_ug_m3",
            "error_pm10_ug_m3": "pm10_error_ug_m3",
            "absolute_error_pm10_ug_m3": "pm10_absolute_error_ug_m3",
        }
    )
    combined["temperature_relative_error_percent"] = (
        combined["temperature_absolute_error_c"] / combined["actual_temperature_c"].abs()
    ) * 100
    combined["pm10_relative_error_percent"] = (
        combined["pm10_absolute_error_ug_m3"] / combined["actual_pm10_ug_m3"].abs()
    ) * 100

    columns = [
        "timestamp",
        "actual_temperature_c",
        "predicted_temperature_c",
        "temperature_error_c",
        "temperature_absolute_error_c",
        "temperature_relative_error_percent",
        "actual_pm10_ug_m3",
        "predicted_pm10_ug_m3",
        "pm10_error_ug_m3",
        "pm10_absolute_error_ug_m3",
        "pm10_relative_error_percent",
    ]
    combined[columns].to_csv(OUTPUT_DIR / "test_predictions_combined.csv", index=False)


def save_test_data_with_predictions() -> None:
    test_data = pd.read_csv(OUTPUT_DIR / "test_latest_15_days.csv", parse_dates=["timestamp"])
    predictions = pd.read_csv(OUTPUT_DIR / "test_predictions_combined.csv", parse_dates=["timestamp"])
    output = test_data.merge(predictions, on="timestamp", how="left")
    output.to_csv(OUTPUT_DIR / "test_data_with_predictions.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(MASTER_CSV, parse_dates=["timestamp"])
    data = data.sort_values("timestamp").set_index("timestamp")
    data = add_model_features(data)

    train, test, test_start = split_latest_15_days(data)

    data.reset_index(names="timestamp").to_csv(OUTPUT_DIR / "merged_hourly_weather_pm10_dataset.csv", index=False)
    train.reset_index(names="timestamp").to_csv(OUTPUT_DIR / "train_latest_15_days_excluded.csv", index=False)
    test.reset_index(names="timestamp").to_csv(OUTPUT_DIR / "test_latest_15_days.csv", index=False)

    metrics = [
        train_and_predict(
            train,
            test,
            "temperature",
            feature_columns_for_target(TARGETS["temperature"]["column"]),
        ),
        train_and_predict(
            train,
            test,
            "pm10",
            feature_columns_for_target(TARGETS["pm10"]["column"]),
        ),
    ]
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    save_combined_predictions()
    save_test_data_with_predictions()

    summary = [
        f"source_csv={MASTER_CSV.name}",
        "model=HistGradientBoostingRegressor",
        "feature_rule=exclude_current_target_column",
        f"test_days={TEST_DAYS}",
        f"dataset_rows={len(data)}",
        f"dataset_start={data.index.min()}",
        f"dataset_end={data.index.max()}",
        f"test_start={test_start}",
        f"training_rows={len(train)}",
        f"testing_rows={len(test)}",
        "",
        "output_files",
        "merged_hourly_weather_pm10_dataset.csv",
        "train_latest_15_days_excluded.csv",
        "test_latest_15_days.csv",
        "model_metrics.csv",
        "predictions_temperature.csv",
        "predictions_pm10.csv",
        "test_predictions_combined.csv",
        "test_data_with_predictions.csv",
    ]
    (OUTPUT_DIR / "README_model_outputs.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n".join(summary))
    print("\nmetrics")
    print(metrics_df.to_string(index=False))
    print("\ntemperature_predictions_head")
    print(pd.read_csv(OUTPUT_DIR / "predictions_temperature.csv").head(10).to_string(index=False))
    print("\npm10_predictions_head")
    print(pd.read_csv(OUTPUT_DIR / "predictions_pm10.csv").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
