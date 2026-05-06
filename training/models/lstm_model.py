"""LSTM training and recursive forecasting."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from training.metrics import regression_metrics
from training.models.common import ModelResult
from utils.config import settings
from utils.model_io import state_model_dir


MODEL_NAME = "lstm"


def train_lstm(train_df: pd.DataFrame, val_df: pd.DataFrame, state: str) -> ModelResult:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    tf.keras.utils.set_random_seed(settings.random_state)

    sequence_length = settings.lstm_sequence_length
    hyperparameters = {
        "sequence_length": sequence_length,
        "lstm_units_1": 64,
        "lstm_units_2": 32,
        "dropout": 0.2,
        "dense_units": 1,
        "optimizer": "adam",
        "loss": "mse",
        "epochs": 50,
        "batch_size": 32,
        "validation_split": 0.1,
        "early_stopping_patience": 5,
        "random_state": settings.random_state,
    }
    scaler = MinMaxScaler()
    scaled_sales = scaler.fit_transform(train_df[["sales"]]).astype("float32")
    x_train, y_train = _make_sequences(scaled_sales, sequence_length)
    if len(x_train) == 0:
        raise ValueError("Not enough rows to train LSTM sequence model.")

    model = Sequential(
        [
            LSTM(
                hyperparameters["lstm_units_1"],
                input_shape=(sequence_length, 1),
                return_sequences=True,
            ),
            Dropout(hyperparameters["dropout"]),
            LSTM(hyperparameters["lstm_units_2"]),
            Dense(hyperparameters["dense_units"]),
        ]
    )
    model.compile(optimizer=hyperparameters["optimizer"], loss=hyperparameters["loss"])
    validation_split = hyperparameters["validation_split"] if len(x_train) > 20 else 0.0
    hyperparameters["effective_validation_split"] = validation_split
    model.fit(
        x_train,
        y_train,
        epochs=hyperparameters["epochs"],
        batch_size=hyperparameters["batch_size"],
        verbose=0,
        validation_split=validation_split,
        callbacks=[
            EarlyStopping(
                patience=hyperparameters["early_stopping_patience"],
                restore_best_weights=True,
            )
        ],
    )

    predictions = recursive_predict_lstm(model, scaler, train_df, len(val_df), sequence_length)
    metrics = regression_metrics(val_df["sales"], predictions)
    artifact_path = _save_lstm_artifacts(model, scaler, state)
    return ModelResult(MODEL_NAME, predictions, metrics, artifact_path, hyperparameters)


def load_lstm_artifacts(state: str):
    import tensorflow as tf

    path = state_model_dir(state) / MODEL_NAME
    model_path = path / "model.keras"
    scaler_path = path / "scaler.joblib"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"LSTM artifacts not found in {path}")
    return tf.keras.models.load_model(model_path), joblib.load(scaler_path)


def recursive_predict_lstm(
    model,
    scaler: MinMaxScaler,
    history: pd.DataFrame,
    horizon_days: int,
    sequence_length: int | None = None,
) -> np.ndarray:
    sequence_length = sequence_length or settings.lstm_sequence_length
    values = history["sales"].to_numpy(dtype="float32").reshape(-1, 1)
    scaled_values = scaler.transform(values).reshape(-1).tolist()
    predictions = []

    for _ in range(horizon_days):
        window = np.asarray(scaled_values[-sequence_length:], dtype="float32")
        if len(window) < sequence_length:
            raise ValueError("Insufficient history for LSTM prediction.")
        x_input = window.reshape(1, sequence_length, 1)
        scaled_prediction = float(model.predict(x_input, verbose=0)[0][0])
        prediction = float(scaler.inverse_transform([[scaled_prediction]])[0][0])
        prediction = max(0.0, prediction)
        predictions.append(prediction)
        scaled_values.append(scaled_prediction)

    return np.asarray(predictions)


def _make_sequences(values: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for index in range(sequence_length, len(values)):
        x.append(values[index - sequence_length : index])
        y.append(values[index])
    return np.asarray(x, dtype="float32"), np.asarray(y, dtype="float32")


def _save_lstm_artifacts(model, scaler: MinMaxScaler, state: str) -> Path:
    path = state_model_dir(state) / MODEL_NAME
    path.mkdir(parents=True, exist_ok=True)
    model.save(path / "model.keras")
    joblib.dump(scaler, path / "scaler.joblib")
    return path
