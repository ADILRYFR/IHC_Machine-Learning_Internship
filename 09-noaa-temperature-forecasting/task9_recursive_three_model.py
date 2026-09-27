from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "task9_noaa_data_and_predictions.csv"
FORECAST_CSV = ROOT / "task9_15_day_forecast.csv"
FORECAST_HOURS = 15 * 24
DEVELOPMENT_CUTOFF = pd.Timestamp("2024-01-01 00:00:00")
VALIDATION_STARTS = pd.to_datetime(
    [
        "2024-01-15 00:00:00",
        "2024-04-15 00:00:00",
        "2024-07-15 00:00:00",
        "2024-10-15 00:00:00",
        "2025-01-15 00:00:00",
        "2025-04-15 00:00:00",
        "2025-07-15 00:00:00",
        "2025-10-15 00:00:00",
    ]
)

SOURCE_COLUMNS = [
    "timestamp_utc",
    "station_id",
    "station_name",
    "latitude",
    "longitude",
    "elevation_m",
    "temperature_c",
    "dew_point_c",
    "relative_humidity_pct",
    "wind_direction_deg",
    "wind_speed_ms",
    "wind_gust_ms",
    "visibility_km",
    "altimeter_hpa",
    "sea_level_pressure_hpa",
    "station_pressure_hpa",
    "ceiling_height_m",
    "precipitation_6h_mm",
    "precipitation_12h_mm",
    "precipitation_24h_mm",
    "observation_count",
    "temperature_observed",
]
BASE_COLUMNS = [
    "temperature_c",
    "dew_point_c",
    "relative_humidity_pct",
    "wind_direction_deg",
    "wind_speed_ms",
    "visibility_km",
    "altimeter_hpa",
]
EXOGENOUS_COLUMNS = [
    "dew_point_c",
    "relative_humidity_pct",
    "wind_direction_deg",
    "wind_speed_ms",
    "visibility_km",
    "altimeter_hpa",
]
TARGET_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
EXOGENOUS_LAGS = [1, 24, 168]

LOOKBACK_HOURS = 24
HIDDEN_SIZE = 16
EPOCHS = 12
BATCH_SIZE = 512
LEARNING_RATE = 0.003
RANDOM_STATE = 42
SEQUENCE_COLUMNS = [
    "temperature_c",
    "dew_point_c",
    "relative_humidity_pct",
    "wind_direction_sin",
    "wind_direction_cos",
    "wind_speed_ms",
    "visibility_km",
    "altimeter_hpa",
    "hour_sin",
    "hour_cos",
    "dayofyear_sin",
    "dayofyear_cos",
]


@dataclass
class ModelResult:
    key: str
    name: str
    validation_metrics: dict[str, float | int]
    test_metrics: dict[str, float | int]
    test_predictions: pd.Series
    future_predictions: pd.Series


class StableLinearRegressor:
    def fit(self, features: pd.DataFrame, target: pd.Series) -> "StableLinearRegressor":
        feature_values = features.to_numpy(dtype=float)
        target_values = target.to_numpy(dtype=float)
        self.feature_offset = feature_values.mean(axis=0)
        self.target_offset = float(target_values.mean())
        centered_features = feature_values - self.feature_offset
        centered_target = target_values - self.target_offset
        condition = max(centered_features.shape) * np.finfo(centered_features.dtype).eps
        self.coefficients = np.linalg.lstsq(
            centered_features,
            centered_target,
            rcond=condition,
        )[0]
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        feature_values = features.to_numpy(dtype=float)
        return (
            (feature_values - self.feature_offset) @ self.coefficients
            + self.target_offset
        )


def load_source() -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(DATA_CSV, parse_dates=["timestamp_utc"])
    if "record_type" in table.columns:
        table = table.loc[table["record_type"] == "historical_observation"].copy()

    missing = [column for column in SOURCE_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"Missing source columns: {missing}")

    source = table[SOURCE_COLUMNS].sort_values("timestamp_utc").copy()
    data = source.set_index("timestamp_utc")
    if data.index.has_duplicates:
        raise ValueError("The CSV contains duplicate historical timestamps")

    expected = pd.date_range(data.index.min(), data.index.max(), freq="h")
    if len(expected) != len(data) or not expected.equals(data.index):
        raise ValueError("The historical data does not contain a complete hourly index")
    return data, source


def training_fill_values(data: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    values = data.loc[data.index < cutoff, BASE_COLUMNS].median(numeric_only=True)
    if values.isna().any():
        missing = values[values.isna()].index.tolist()
        raise ValueError(f"No training median is available for {missing}")
    return values


def prepare_state(data: pd.DataFrame, fill_values: pd.Series) -> pd.DataFrame:
    state = data[BASE_COLUMNS].copy()
    state = state.ffill(limit=6).fillna(fill_values)
    radians = np.deg2rad(state["wind_direction_deg"])
    state["wind_direction_sin"] = np.sin(radians)
    state["wind_direction_cos"] = np.cos(radians)
    return state


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    features = pd.DataFrame(index=index)
    features["hour_sin"] = np.sin(2.0 * np.pi * index.hour / 24.0)
    features["hour_cos"] = np.cos(2.0 * np.pi * index.hour / 24.0)
    features["dayofyear_sin"] = np.sin(2.0 * np.pi * index.dayofyear / 366.0)
    features["dayofyear_cos"] = np.cos(2.0 * np.pi * index.dayofyear / 366.0)
    features["dayofweek_sin"] = np.sin(2.0 * np.pi * index.dayofweek / 7.0)
    features["dayofweek_cos"] = np.cos(2.0 * np.pi * index.dayofweek / 7.0)
    return features


def build_tabular_features(
    data: pd.DataFrame,
    fill_values: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    state = prepare_state(data, fill_values)
    features = calendar_features(state.index)

    for lag in TARGET_LAGS:
        features[f"temperature_lag_{lag}h"] = state["temperature_c"].shift(lag)

    shifted_temperature = state["temperature_c"].shift(1)
    features["temperature_mean_3h"] = shifted_temperature.rolling(3).mean()
    features["temperature_mean_24h"] = shifted_temperature.rolling(24).mean()
    features["temperature_mean_168h"] = shifted_temperature.rolling(168).mean()
    features["temperature_std_24h"] = shifted_temperature.rolling(24).std()

    state_columns = [
        "dew_point_c",
        "relative_humidity_pct",
        "wind_direction_sin",
        "wind_direction_cos",
        "wind_speed_ms",
        "visibility_km",
        "altimeter_hpa",
    ]
    for column in state_columns:
        for lag in EXOGENOUS_LAGS:
            features[f"{column}_lag_{lag}h"] = state[column].shift(lag)
        features[f"{column}_mean_24h"] = state[column].shift(1).rolling(24).mean()

    return features, data["temperature_c"].copy()


def fit_tabular_model(
    data: pd.DataFrame,
    cutoff: pd.Timestamp,
    estimator_factory: Callable[[], object],
) -> tuple[object, pd.Series, list[str]]:
    fill_values = training_fill_values(data, cutoff)
    features, target = build_tabular_features(data, fill_values)
    mask = (features.index < cutoff) & target.notna() & features.notna().all(axis=1)
    model = estimator_factory()
    model.fit(features.loc[mask], target.loc[mask])
    return model, fill_values, features.columns.tolist()


def scalar_feature_row(state: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    values: dict[str, float] = {
        "hour_sin": float(np.sin(2.0 * np.pi * timestamp.hour / 24.0)),
        "hour_cos": float(np.cos(2.0 * np.pi * timestamp.hour / 24.0)),
        "dayofyear_sin": float(np.sin(2.0 * np.pi * timestamp.dayofyear / 366.0)),
        "dayofyear_cos": float(np.cos(2.0 * np.pi * timestamp.dayofyear / 366.0)),
        "dayofweek_sin": float(np.sin(2.0 * np.pi * timestamp.dayofweek / 7.0)),
        "dayofweek_cos": float(np.cos(2.0 * np.pi * timestamp.dayofweek / 7.0)),
    }

    for lag in TARGET_LAGS:
        values[f"temperature_lag_{lag}h"] = float(
            state.at[timestamp - pd.Timedelta(hours=lag), "temperature_c"]
        )

    prior_temperature = state.loc[
        timestamp - pd.Timedelta(hours=168) : timestamp - pd.Timedelta(hours=1),
        "temperature_c",
    ]
    values["temperature_mean_3h"] = float(prior_temperature.iloc[-3:].mean())
    values["temperature_mean_24h"] = float(prior_temperature.iloc[-24:].mean())
    values["temperature_mean_168h"] = float(prior_temperature.mean())
    values["temperature_std_24h"] = float(prior_temperature.iloc[-24:].std())

    state_columns = [
        "dew_point_c",
        "relative_humidity_pct",
        "wind_direction_sin",
        "wind_direction_cos",
        "wind_speed_ms",
        "visibility_km",
        "altimeter_hpa",
    ]
    for column in state_columns:
        for lag in EXOGENOUS_LAGS:
            values[f"{column}_lag_{lag}h"] = float(
                state.at[timestamp - pd.Timedelta(hours=lag), column]
            )
        history = state.loc[
            timestamp - pd.Timedelta(hours=24) : timestamp - pd.Timedelta(hours=1),
            column,
        ]
        values[f"{column}_mean_24h"] = float(history.mean())

    return pd.DataFrame([values], index=[timestamp])


def append_projected_state(
    state: pd.DataFrame,
    timestamp: pd.Timestamp,
    predicted_temperature: float,
) -> None:
    weekly_timestamp = timestamp - pd.Timedelta(hours=168)
    previous_timestamp = timestamp - pd.Timedelta(hours=1)
    row = state.loc[weekly_timestamp, EXOGENOUS_COLUMNS].copy()
    row = row.fillna(state.loc[previous_timestamp, EXOGENOUS_COLUMNS])
    state.loc[timestamp, "temperature_c"] = predicted_temperature
    for column in EXOGENOUS_COLUMNS:
        state.loc[timestamp, column] = row[column]
    radians = np.deg2rad(state.at[timestamp, "wind_direction_deg"])
    state.loc[timestamp, "wind_direction_sin"] = np.sin(radians)
    state.loc[timestamp, "wind_direction_cos"] = np.cos(radians)


def recursive_tabular_forecast(
    model: object,
    data: pd.DataFrame,
    start: pd.Timestamp,
    hours: int,
    fill_values: pd.Series,
    feature_names: list[str],
) -> pd.Series:
    state = prepare_state(data.loc[data.index < start], fill_values)
    timestamps = pd.date_range(start, periods=hours, freq="h")
    predictions: list[float] = []

    for timestamp in timestamps:
        row = scalar_feature_row(state, timestamp)[feature_names]
        prediction = float(model.predict(row)[0])
        predictions.append(prediction)
        append_projected_state(state, timestamp, prediction)

    return pd.Series(predictions, index=timestamps, name="predicted_temperature_c")


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    joined = pd.concat(
        [actual.rename("actual"), predicted.rename("predicted")], axis=1
    ).dropna()
    error = joined["predicted"] - joined["actual"]
    relative = np.abs(error) / joined["actual"].abs().replace(0, np.nan) * 100.0
    mse = mean_squared_error(joined["actual"], joined["predicted"])
    return {
        "rows": int(len(joined)),
        "mae": float(mean_absolute_error(joined["actual"], joined["predicted"])),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(joined["actual"], joined["predicted"])),
        "mean_relative_error_pct": float(relative.mean()),
    }


def evaluate_tabular_model(
    key: str,
    name: str,
    estimator_factory: Callable[[], object],
    data: pd.DataFrame,
) -> ModelResult:
    development_model, development_fill, feature_names = fit_tabular_model(
        data, DEVELOPMENT_CUTOFF, estimator_factory
    )
    validation_actual: list[pd.Series] = []
    validation_predicted: list[pd.Series] = []
    for start in VALIDATION_STARTS:
        predicted = recursive_tabular_forecast(
            development_model,
            data,
            start,
            FORECAST_HOURS,
            development_fill,
            feature_names,
        )
        validation_predicted.append(predicted)
        validation_actual.append(data["temperature_c"].reindex(predicted.index))

    validation_metrics = calculate_metrics(
        pd.concat(validation_actual), pd.concat(validation_predicted)
    )
    latest = data.index.max()
    test_start = latest - pd.Timedelta(hours=FORECAST_HOURS - 1)

    test_model, test_fill, test_features = fit_tabular_model(
        data, test_start, estimator_factory
    )
    test_predictions = recursive_tabular_forecast(
        test_model,
        data,
        test_start,
        FORECAST_HOURS,
        test_fill,
        test_features,
    )
    test_actual = data["temperature_c"].reindex(test_predictions.index)
    test_metrics = calculate_metrics(test_actual, test_predictions)

    future_start = latest + pd.Timedelta(hours=1)
    future_model, future_fill, future_features = fit_tabular_model(
        data, future_start, estimator_factory
    )
    future_predictions = recursive_tabular_forecast(
        future_model,
        data,
        future_start,
        FORECAST_HOURS,
        future_fill,
        future_features,
    )
    return ModelResult(
        key,
        name,
        validation_metrics,
        test_metrics,
        test_predictions,
        future_predictions,
    )


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


class LSTMRegressor:
    def __init__(self, input_size: int, hidden_size: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        input_scale = np.sqrt(1.0 / (input_size + hidden_size))
        output_scale = np.sqrt(1.0 / hidden_size)
        self.params = {
            "Wx": rng.normal(0, input_scale, (input_size, hidden_size * 4)).astype(np.float32),
            "Wh": rng.normal(0, input_scale, (hidden_size, hidden_size * 4)).astype(np.float32),
            "b": np.zeros(hidden_size * 4, dtype=np.float32),
            "Wy": rng.normal(0, output_scale, (hidden_size, 1)).astype(np.float32),
            "by": np.zeros(1, dtype=np.float32),
        }
        self.params["b"][hidden_size : hidden_size * 2] = 1.0
        self.hidden_size = hidden_size
        self.m = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.v = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.step_count = 0

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, ...]]]:
        batch_size, steps, _ = inputs.shape
        hidden = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        cell = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        cache: list[tuple[np.ndarray, ...]] = []

        for step in range(steps):
            current = inputs[:, step, :]
            previous_hidden = hidden
            previous_cell = cell
            gates = (
                current @ self.params["Wx"]
                + previous_hidden @ self.params["Wh"]
                + self.params["b"]
            )
            input_gate = sigmoid(gates[:, : self.hidden_size])
            forget_gate = sigmoid(gates[:, self.hidden_size : self.hidden_size * 2])
            candidate = np.tanh(gates[:, self.hidden_size * 2 : self.hidden_size * 3])
            output_gate = sigmoid(gates[:, self.hidden_size * 3 :])
            cell = forget_gate * previous_cell + input_gate * candidate
            hidden = output_gate * np.tanh(cell)
            cache.append(
                (
                    current,
                    previous_hidden,
                    previous_cell,
                    input_gate,
                    forget_gate,
                    candidate,
                    output_gate,
                    cell,
                    hidden,
                )
            )

        prediction = hidden @ self.params["Wy"] + self.params["by"]
        return prediction[:, 0], cache

    def update(self, inputs: np.ndarray, target: np.ndarray, learning_rate: float) -> float:
        prediction, cache = self.forward(inputs)
        residual = prediction - target
        loss = float(np.mean(residual**2))
        batch_size = len(target)
        gradients = {key: np.zeros_like(value) for key, value in self.params.items()}

        output_gradient = (2.0 / batch_size) * residual.reshape(-1, 1)
        final_hidden = cache[-1][-1]
        gradients["Wy"] = final_hidden.T @ output_gradient
        gradients["by"] = output_gradient.sum(axis=0)
        hidden_gradient = output_gradient @ self.params["Wy"].T
        cell_gradient = np.zeros_like(hidden_gradient)

        for values in reversed(cache):
            (
                current,
                previous_hidden,
                previous_cell,
                input_gate,
                forget_gate,
                candidate,
                output_gate,
                cell,
                _,
            ) = values
            tanh_cell = np.tanh(cell)
            output_raw = hidden_gradient * tanh_cell * output_gate * (1.0 - output_gate)
            combined_cell = hidden_gradient * output_gate * (1.0 - tanh_cell**2) + cell_gradient
            forget_raw = combined_cell * previous_cell * forget_gate * (1.0 - forget_gate)
            input_raw = combined_cell * candidate * input_gate * (1.0 - input_gate)
            candidate_raw = combined_cell * input_gate * (1.0 - candidate**2)
            gate_gradient = np.concatenate(
                [input_raw, forget_raw, candidate_raw, output_raw], axis=1
            )
            gradients["Wx"] += current.T @ gate_gradient
            gradients["Wh"] += previous_hidden.T @ gate_gradient
            gradients["b"] += gate_gradient.sum(axis=0)
            hidden_gradient = gate_gradient @ self.params["Wh"].T
            cell_gradient = combined_cell * forget_gate

        self.step_count += 1
        for key, gradient in gradients.items():
            np.clip(gradient, -1.0, 1.0, out=gradient)
            self.m[key] = 0.9 * self.m[key] + 0.1 * gradient
            self.v[key] = 0.999 * self.v[key] + 0.001 * gradient**2
            adjusted_m = self.m[key] / (1.0 - 0.9**self.step_count)
            adjusted_v = self.v[key] / (1.0 - 0.999**self.step_count)
            self.params[key] -= learning_rate * adjusted_m / (np.sqrt(adjusted_v) + 1e-8)
        return loss

    def predict(self, inputs: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        predictions: list[np.ndarray] = []
        for start in range(0, len(inputs), batch_size):
            prediction, _ = self.forward(inputs[start : start + batch_size])
            predictions.append(prediction)
        return np.concatenate(predictions)


def sequence_state(data: pd.DataFrame, fill_values: pd.Series) -> pd.DataFrame:
    state = prepare_state(data, fill_values)
    index = state.index
    state["hour_sin"] = np.sin(2.0 * np.pi * index.hour / 24.0)
    state["hour_cos"] = np.cos(2.0 * np.pi * index.hour / 24.0)
    state["dayofyear_sin"] = np.sin(2.0 * np.pi * index.dayofyear / 366.0)
    state["dayofyear_cos"] = np.cos(2.0 * np.pi * index.dayofyear / 366.0)
    return state


def fit_lstm(
    data: pd.DataFrame,
    cutoff: pd.Timestamp,
    seed: int,
) -> tuple[LSTMRegressor, pd.Series, np.ndarray, np.ndarray, float, float]:
    fill_values = training_fill_values(data, cutoff)
    state = sequence_state(data, fill_values)
    raw_values = state[SEQUENCE_COLUMNS].to_numpy(dtype=np.float32)
    training_rows = state.index < cutoff
    feature_mean = raw_values[training_rows].mean(axis=0)
    feature_std = raw_values[training_rows].std(axis=0)
    feature_std[feature_std == 0] = 1.0
    scaled_values = (raw_values - feature_mean) / feature_std

    target = data["temperature_c"].to_numpy(dtype=np.float32)
    positions = np.arange(LOOKBACK_HOURS, len(data))
    valid = (data.index[positions] < cutoff) & np.isfinite(target[positions])
    target_positions = positions[valid]
    target_mean = float(target[target_positions].mean())
    target_std = float(target[target_positions].std()) or 1.0
    scaled_target = (target[target_positions] - target_mean) / target_std

    windows = np.lib.stride_tricks.sliding_window_view(
        scaled_values, window_shape=LOOKBACK_HOURS, axis=0
    ).transpose(0, 2, 1)
    window_positions = target_positions - LOOKBACK_HOURS
    model = LSTMRegressor(len(SEQUENCE_COLUMNS), HIDDEN_SIZE, seed)
    rng = np.random.default_rng(seed)

    for epoch in range(EPOCHS):
        order = rng.permutation(len(window_positions))
        losses: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            inputs = np.ascontiguousarray(windows[window_positions[batch]], dtype=np.float32)
            losses.append(model.update(inputs, scaled_target[batch], LEARNING_RATE))
        print(f"lstm_epoch={epoch + 1}/{EPOCHS} loss={np.mean(losses):.6f}")

    return model, fill_values, feature_mean, feature_std, target_mean, target_std


def recursive_lstm_forecast(
    bundle: tuple[LSTMRegressor, pd.Series, np.ndarray, np.ndarray, float, float],
    data: pd.DataFrame,
    start: pd.Timestamp,
) -> pd.Series:
    model, fill_values, feature_mean, feature_std, target_mean, target_std = bundle
    state = sequence_state(data.loc[data.index < start], fill_values)
    timestamps = pd.date_range(start, periods=FORECAST_HOURS, freq="h")
    predictions: list[float] = []

    for timestamp in timestamps:
        sequence = state.iloc[-LOOKBACK_HOURS:][SEQUENCE_COLUMNS].to_numpy(dtype=np.float32)
        scaled = (sequence - feature_mean) / feature_std
        predicted_scaled = float(model.predict(scaled[None, :, :])[0])
        prediction = predicted_scaled * target_std + target_mean
        predictions.append(prediction)
        append_projected_state(state, timestamp, prediction)
        state.loc[timestamp, "hour_sin"] = np.sin(2.0 * np.pi * timestamp.hour / 24.0)
        state.loc[timestamp, "hour_cos"] = np.cos(2.0 * np.pi * timestamp.hour / 24.0)
        state.loc[timestamp, "dayofyear_sin"] = np.sin(
            2.0 * np.pi * timestamp.dayofyear / 366.0
        )
        state.loc[timestamp, "dayofyear_cos"] = np.cos(
            2.0 * np.pi * timestamp.dayofyear / 366.0
        )

    return pd.Series(predictions, index=timestamps, name="predicted_temperature_c")


def evaluate_lstm_model(data: pd.DataFrame) -> ModelResult:
    development_bundle = fit_lstm(data, DEVELOPMENT_CUTOFF, RANDOM_STATE)
    validation_actual: list[pd.Series] = []
    validation_predicted: list[pd.Series] = []
    for start in VALIDATION_STARTS:
        predicted = recursive_lstm_forecast(development_bundle, data, start)
        validation_predicted.append(predicted)
        validation_actual.append(data["temperature_c"].reindex(predicted.index))
    validation_metrics = calculate_metrics(
        pd.concat(validation_actual), pd.concat(validation_predicted)
    )
    del development_bundle

    latest = data.index.max()
    test_start = latest - pd.Timedelta(hours=FORECAST_HOURS - 1)
    test_bundle = fit_lstm(data, test_start, RANDOM_STATE + 1)
    test_predictions = recursive_lstm_forecast(test_bundle, data, test_start)
    test_actual = data["temperature_c"].reindex(test_predictions.index)
    test_metrics = calculate_metrics(test_actual, test_predictions)
    del test_bundle

    future_start = latest + pd.Timedelta(hours=1)
    future_bundle = fit_lstm(data, future_start, RANDOM_STATE + 2)
    future_predictions = recursive_lstm_forecast(future_bundle, data, future_start)
    return ModelResult(
        "lstm_rnn",
        "LSTM Recurrent Neural Network",
        validation_metrics,
        test_metrics,
        test_predictions,
        future_predictions,
    )


def make_linear_model() -> StableLinearRegressor:
    return StableLinearRegressor()


def make_gradient_boosting_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=350,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=42,
    )


def write_output_csvs(
    source: pd.DataFrame,
    results: list[ModelResult],
) -> None:
    history = source.copy().set_index("timestamp_utc")
    history["record_type"] = "historical_observation"
    history["prediction_stage"] = "historical_only"
    history["actual_temperature_c"] = history["temperature_c"]

    prediction_columns: list[str] = []
    for result in results:
        predicted_column = f"{result.key}_predicted_temperature_c"
        absolute_column = f"{result.key}_absolute_error_c"
        relative_column = f"{result.key}_relative_error_pct"
        prediction_columns.extend([predicted_column, absolute_column, relative_column])
        history[predicted_column] = np.nan
        history[absolute_column] = np.nan
        history[relative_column] = np.nan
        history.loc[result.test_predictions.index, predicted_column] = result.test_predictions
        actual = history.loc[result.test_predictions.index, "actual_temperature_c"]
        absolute_error = (result.test_predictions - actual).abs()
        history.loc[result.test_predictions.index, absolute_column] = absolute_error
        history.loc[result.test_predictions.index, relative_column] = (
            absolute_error / actual.abs().replace(0, np.nan) * 100.0
        )

    test_index = results[0].test_predictions.index
    history.loc[test_index, "prediction_stage"] = "final_15_day_test"

    history = history.reset_index(names="timestamp_utc")
    for column in prediction_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce").round(4)
    output_columns = SOURCE_COLUMNS + [
        "record_type",
        "prediction_stage",
        "actual_temperature_c",
        *prediction_columns,
    ]
    history[output_columns].to_csv(
        DATA_CSV,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )

    future_index = results[0].future_predictions.index
    latest = history.iloc[-1]
    future = pd.DataFrame(index=future_index)
    for column in ["station_id", "station_name", "latitude", "longitude", "elevation_m"]:
        future[column] = latest[column]
    future["forecast_horizon_hour"] = np.arange(1, len(future) + 1)
    future["record_type"] = "future_forecast"
    future["prediction_stage"] = "future_15_day_forecast"
    forecast_prediction_columns: list[str] = []
    for result in results:
        predicted_column = f"{result.key}_predicted_temperature_c"
        forecast_prediction_columns.append(predicted_column)
        future[predicted_column] = result.future_predictions.round(4)

    forecast_columns = [
        "timestamp_utc",
        "station_id",
        "station_name",
        "latitude",
        "longitude",
        "elevation_m",
        "forecast_horizon_hour",
        "record_type",
        "prediction_stage",
        *forecast_prediction_columns,
    ]
    future.reset_index(names="timestamp_utc")[forecast_columns].to_csv(
        FORECAST_CSV,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )


def print_metrics(results: list[ModelResult]) -> None:
    print("\nModel comparison")
    for result in results:
        validation = result.validation_metrics
        test = result.test_metrics
        print(f"\n{result.name}")
        print(
            "validation: "
            f"MAE={validation['mae']:.3f}, "
            f"RMSE={validation['rmse']:.3f}, "
            f"R2={validation['r2']:.3f}"
        )
        print(
            "final test: "
            f"MAE={test['mae']:.3f}, "
            f"MSE={test['mse']:.3f}, "
            f"RMSE={test['rmse']:.3f}, "
            f"R2={test['r2']:.3f}, "
            f"mean relative error={test['mean_relative_error_pct']:.2f}%"
        )
    print(f"\nUpdated data CSV: {DATA_CSV.name}")
    print(f"Updated forecast CSV: {FORECAST_CSV.name}")


def main() -> None:
    data, source = load_source()
    results = [
        evaluate_tabular_model(
            "multiple_linear_regression",
            "Multiple Linear Regression",
            make_linear_model,
            data,
        ),
        evaluate_tabular_model(
            "gradient_boosting",
            "Histogram Gradient Boosting",
            make_gradient_boosting_model,
            data,
        ),
        evaluate_lstm_model(data),
    ]
    write_output_csvs(source, results)
    print_metrics(results)


if __name__ == "__main__":
    main()
