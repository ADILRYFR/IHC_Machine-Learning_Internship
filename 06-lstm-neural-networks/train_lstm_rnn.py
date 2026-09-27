from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_CSV = Path(__file__).resolve().parent / "task6_weather_pm10_hourly_master.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "task6_lstm_rnn_predictions.csv"

LOOKBACK_HOURS = 24
TEST_DAYS = 15
RANDOM_STATE = 42


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def add_features(data: pd.DataFrame) -> pd.DataFrame:
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


def make_sequences(data: pd.DataFrame, target_column: str, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    ready = data.dropna(subset=feature_columns + [target_column]).copy()
    features = ready[feature_columns].to_numpy(dtype=float)
    target = ready[target_column].to_numpy(dtype=float)
    timestamps = ready.index

    sequence_count = len(ready) - LOOKBACK_HOURS + 1
    X = np.empty((sequence_count, LOOKBACK_HOURS, len(feature_columns)), dtype=float)
    y = np.empty(sequence_count, dtype=float)
    target_times = []

    for row in range(sequence_count):
        end = row + LOOKBACK_HOURS
        X[row] = features[row:end]
        y[row] = target[end - 1]
        target_times.append(timestamps[end - 1])

    return X, y, pd.DatetimeIndex(target_times)


class LSTMRegressor:
    def __init__(self, input_size: int, hidden_size: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        scale_x = np.sqrt(1.0 / (input_size + hidden_size))
        scale_y = np.sqrt(1.0 / hidden_size)

        self.params = {
            "Wx": rng.normal(0, scale_x, (input_size, hidden_size * 4)),
            "Wh": rng.normal(0, scale_x, (hidden_size, hidden_size * 4)),
            "b": np.zeros(hidden_size * 4),
            "Wy": rng.normal(0, scale_y, (hidden_size, 1)),
            "by": np.zeros(1),
        }
        self.params["b"][hidden_size : hidden_size * 2] = 1.0
        self.hidden_size = hidden_size
        self.m = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.v = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.step_count = 0

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, ...]]]:
        batch_size, steps, _ = X.shape
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))
        cache = []

        for t in range(steps):
            x_t = X[:, t, :]
            h_prev = h
            c_prev = c
            z = x_t @ self.params["Wx"] + h_prev @ self.params["Wh"] + self.params["b"]
            i = sigmoid(z[:, : self.hidden_size])
            f = sigmoid(z[:, self.hidden_size : self.hidden_size * 2])
            g = np.tanh(z[:, self.hidden_size * 2 : self.hidden_size * 3])
            o = sigmoid(z[:, self.hidden_size * 3 :])
            c = f * c_prev + i * g
            h = o * np.tanh(c)
            cache.append((x_t, h_prev, c_prev, i, f, g, o, c, h))

        y_hat = h @ self.params["Wy"] + self.params["by"]
        return y_hat[:, 0], cache

    def update(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> float:
        y_hat, cache = self.forward(X)
        batch_size = len(y)
        residual = y_hat - y
        loss = float(np.mean(residual**2))

        grads = {key: np.zeros_like(value) for key, value in self.params.items()}
        dy = (2.0 / batch_size) * residual.reshape(-1, 1)
        h_final = cache[-1][-1]
        grads["Wy"] = h_final.T @ dy
        grads["by"] = dy.sum(axis=0)

        dh_next = dy @ self.params["Wy"].T
        dc_next = np.zeros_like(dh_next)

        for values in reversed(cache):
            x_t, h_prev, c_prev, i, f, g, o, c, _ = values
            tanh_c = np.tanh(c)
            do = dh_next * tanh_c
            do_raw = do * o * (1.0 - o)

            dc = dh_next * o * (1.0 - tanh_c**2) + dc_next
            df_raw = (dc * c_prev) * f * (1.0 - f)
            di_raw = (dc * g) * i * (1.0 - i)
            dg_raw = (dc * i) * (1.0 - g**2)

            dz = np.concatenate([di_raw, df_raw, dg_raw, do_raw], axis=1)
            grads["Wx"] += x_t.T @ dz
            grads["Wh"] += h_prev.T @ dz
            grads["b"] += dz.sum(axis=0)

            dh_next = dz @ self.params["Wh"].T
            dc_next = dc * f

        self.step_count += 1
        for key in grads:
            np.clip(grads[key], -1.0, 1.0, out=grads[key])
            self.m[key] = 0.9 * self.m[key] + 0.1 * grads[key]
            self.v[key] = 0.999 * self.v[key] + 0.001 * (grads[key] ** 2)
            m_hat = self.m[key] / (1.0 - 0.9**self.step_count)
            v_hat = self.v[key] / (1.0 - 0.999**self.step_count)
            self.params[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

        return loss

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        values = []
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            y_hat, _ = self.forward(X[start:end])
            values.append(y_hat)
        return np.concatenate(values)


def standardize(train_X: np.ndarray, test_X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_X.reshape(-1, train_X.shape[-1]).mean(axis=0)
    std = train_X.reshape(-1, train_X.shape[-1]).std(axis=0)
    std[std == 0] = 1.0
    return (train_X - mean) / std, (test_X - mean) / std


def train_target(data: pd.DataFrame, target_column: str, seed: int) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    test_start = data.index.max() - pd.Timedelta(days=TEST_DAYS)
    feature_columns = feature_columns_for_target(target_column)
    X, y, timestamps = make_sequences(data, target_column, feature_columns)

    train_mask = timestamps < test_start
    test_mask = timestamps >= test_start

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    test_times = timestamps[test_mask]

    X_train, X_test = standardize(X_train, X_test)
    y_mean = y_train.mean()
    y_std = y_train.std() if y_train.std() != 0 else 1.0
    y_train_scaled = (y_train - y_mean) / y_std

    model = LSTMRegressor(input_size=X_train.shape[-1], hidden_size=16, seed=seed)
    rng = np.random.default_rng(seed)
    batch_size = 512
    epochs = 20
    learning_rate = 0.003

    for _ in range(epochs):
        order = rng.permutation(len(X_train))
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            model.update(X_train[rows], y_train_scaled[rows], learning_rate)

    predictions = model.predict(X_test) * y_std + y_mean
    return pd.Series(test_times), y_test, predictions


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    mse = float(np.mean(error**2))
    rmse = mse**0.5
    mae = float(np.mean(np.abs(error)))
    denominator = np.sum((actual - actual.mean()) ** 2)
    r2 = 1.0 - (np.sum(error**2) / denominator)
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": float(r2)}


def main() -> None:
    data = pd.read_csv(SOURCE_CSV, parse_dates=["timestamp"])
    data = data.sort_values("timestamp").set_index("timestamp")
    data = add_features(data)

    times, actual_temperature, predicted_temperature = train_target(
        data,
        "air_temperature_c",
        RANDOM_STATE,
    )
    _, actual_pm10, predicted_pm10 = train_target(
        data,
        "pm10_outdoors_ug_m3",
        RANDOM_STATE + 1,
    )

    output = pd.DataFrame(
        {
            "timestamp": times,
            "actual_temperature_c": actual_temperature,
            "predicted_temperature_c": predicted_temperature,
            "actual_pm10_ug_m3": actual_pm10,
            "predicted_pm10_ug_m3": predicted_pm10,
        }
    )
    output["temperature_error_c"] = output["predicted_temperature_c"] - output["actual_temperature_c"]
    output["temperature_absolute_error_c"] = output["temperature_error_c"].abs()
    output["temperature_relative_error_percent"] = (
        output["temperature_absolute_error_c"] / output["actual_temperature_c"].abs()
    ) * 100
    output["pm10_error_ug_m3"] = output["predicted_pm10_ug_m3"] - output["actual_pm10_ug_m3"]
    output["pm10_absolute_error_ug_m3"] = output["pm10_error_ug_m3"].abs()
    output["pm10_relative_error_percent"] = (
        output["pm10_absolute_error_ug_m3"] / output["actual_pm10_ug_m3"].abs()
    ) * 100

    output.to_csv(OUTPUT_CSV, index=False)

    temperature_metrics = metrics(actual_temperature, predicted_temperature)
    pm10_metrics = metrics(actual_pm10, predicted_pm10)

    print("model=LSTM_RNN")
    print(f"source_csv={SOURCE_CSV.name}")
    print(f"output_csv={OUTPUT_CSV.name}")
    print(f"test_rows={len(output)}")
    print(
        "temperature "
        f"mae={temperature_metrics['mae']:.6f} "
        f"mse={temperature_metrics['mse']:.6f} "
        f"rmse={temperature_metrics['rmse']:.6f} "
        f"r2={temperature_metrics['r2']:.6f}"
    )
    print(
        "pm10 "
        f"mae={pm10_metrics['mae']:.6f} "
        f"mse={pm10_metrics['mse']:.6f} "
        f"rmse={pm10_metrics['rmse']:.6f} "
        f"r2={pm10_metrics['r2']:.6f}"
    )


if __name__ == "__main__":
    main()
