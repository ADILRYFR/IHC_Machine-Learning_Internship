from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_CSV = Path(__file__).resolve().parent / "task7_weather_pm10_hourly_master.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "june_forecast_lstm_rnn.csv"

LOOKBACK_HOURS = 24
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


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


def make_sequences(data: pd.DataFrame, target_column: str) -> tuple[np.ndarray, np.ndarray]:
    ready = data.dropna(subset=FEATURE_COLUMNS + [target_column]).copy()
    features = ready[FEATURE_COLUMNS].to_numpy(dtype=float)
    target = ready[target_column].to_numpy(dtype=float)

    sequence_count = len(ready) - LOOKBACK_HOURS + 1
    X = np.empty((sequence_count, LOOKBACK_HOURS, len(FEATURE_COLUMNS)), dtype=float)
    y = np.empty(sequence_count, dtype=float)

    for row in range(sequence_count):
        end = row + LOOKBACK_HOURS
        X[row] = features[row:end]
        y[row] = target[end - 1]

    return X, y


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


def standardize_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    std = X.reshape(-1, X.shape[-1]).std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def train_target(data: pd.DataFrame, target_column: str, seed: int) -> tuple[LSTMRegressor, np.ndarray, np.ndarray, float, float]:
    X, y = make_sequences(data, target_column)
    X_train, feature_mean, feature_std = standardize_features(X)

    y_mean = float(y.mean())
    y_std = float(y.std()) if y.std() != 0 else 1.0
    y_train = (y - y_mean) / y_std

    model = LSTMRegressor(input_size=X_train.shape[-1], hidden_size=16, seed=seed)
    rng = np.random.default_rng(seed)
    batch_size = 512
    epochs = 20
    learning_rate = 0.003

    for _ in range(epochs):
        order = rng.permutation(len(X_train))
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            model.update(X_train[rows], y_train[rows], learning_rate)

    return model, feature_mean, feature_std, y_mean, y_std


def forecast_june(
    data: pd.DataFrame,
    temperature_model: LSTMRegressor,
    pm10_model: LSTMRegressor,
    temperature_scale: tuple[np.ndarray, np.ndarray, float, float],
    pm10_scale: tuple[np.ndarray, np.ndarray, float, float],
) -> pd.DataFrame:
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
        ready = featured.dropna(subset=FEATURE_COLUMNS)
        sequence = ready[FEATURE_COLUMNS].tail(LOOKBACK_HOURS).to_numpy(dtype=float)
        X_next = sequence.reshape(1, LOOKBACK_HOURS, len(FEATURE_COLUMNS))

        temp_mean, temp_std, temp_y_mean, temp_y_std = temperature_scale
        pm10_mean, pm10_std, pm10_y_mean, pm10_y_std = pm10_scale

        X_temp = (X_next - temp_mean) / temp_std
        X_pm10 = (X_next - pm10_mean) / pm10_std

        predicted_temperature = float(temperature_model.predict(X_temp)[0] * temp_y_std + temp_y_mean)
        predicted_pm10 = float(pm10_model.predict(X_pm10)[0] * pm10_y_std + pm10_y_mean)

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

    featured = add_forecast_features(data)
    temperature_model, temp_mean, temp_std, temp_y_mean, temp_y_std = train_target(
        featured,
        "air_temperature_c",
        RANDOM_STATE,
    )
    pm10_model, pm10_mean, pm10_std, pm10_y_mean, pm10_y_std = train_target(
        featured,
        "pm10_outdoors_ug_m3",
        RANDOM_STATE + 1,
    )

    forecast = forecast_june(
        data,
        temperature_model,
        pm10_model,
        (temp_mean, temp_std, temp_y_mean, temp_y_std),
        (pm10_mean, pm10_std, pm10_y_mean, pm10_y_std),
    )
    forecast.to_csv(OUTPUT_CSV, index=False)

    print("model=LSTM_RNN")
    print(f"source_csv={SOURCE_CSV.name}")
    print(f"output_csv={OUTPUT_CSV.name}")
    print(f"source_start={data.index.min()}")
    print(f"source_end={data.index.max()}")
    print(f"forecast_start={forecast['timestamp'].min()}")
    print(f"forecast_end={forecast['timestamp'].max()}")
    print(f"forecast_rows={len(forecast)}")


if __name__ == "__main__":
    main()
