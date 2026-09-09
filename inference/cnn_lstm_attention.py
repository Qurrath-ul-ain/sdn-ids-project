"""Final CNN-LSTM-Attention inference interface.

Loads the final .keras artifact by reconstructing the model architecture
and reading its HDF5 weights directly. This avoids Keras deserialization
issues with the legacy `time_major` argument.
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "models"
    / "final_cicflow_temporal"
    / "cnn_lstm_attention_final.keras"
)

METADATA_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "final_temporal_sequence_metadata.json"
)

SCALER_PATH = (
    ROOT
    / "data"
    / "processed"
    / "final_cicflow_temporal"
    / "scaler.json"
)


class Attention(tf.keras.layers.Layer):
    """Temporal attention layer used by the final trained model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = tf.keras.layers.Dense(1)

    def call(self, inputs):
        weights = tf.nn.softmax(self.score(inputs), axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)

    def get_config(self):
        return super().get_config()


def build_model() -> tf.keras.Model:
    """Reconstruct the exact final model architecture."""

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(10, 5),
                name="input_temporal_window",
            ),
            tf.keras.layers.Conv1D(
                32,
                3,
                padding="same",
                activation="relu",
                name="conv1d_32",
            ),
            tf.keras.layers.MaxPooling1D(
                2,
                name="max_pooling1d",
            ),
            tf.keras.layers.LSTM(
                64,
                return_sequences=True,
                name="lstm_64",
            ),
            Attention(
                name="attention_layer",
            ),
            tf.keras.layers.Dense(
                32,
                activation="relu",
                name="dense_32",
            ),
            tf.keras.layers.Dropout(
                0.3,
                name="dropout_03",
            ),
            tf.keras.layers.Dense(
                4,
                activation="softmax",
                name="dense_softmax",
            ),
        ],
        name="CNN_LSTM_Attention_Final",
    )


def load_weights_from_keras_archive(
    model: tf.keras.Model,
    model_path: Path,
) -> None:
    """Read model.weights.h5 from the .keras archive."""

    with zipfile.ZipFile(model_path, "r") as archive:
        with archive.open("model.weights.h5") as weights_file:
            with tempfile.NamedTemporaryFile(
                suffix=".h5",
                delete=False,
            ) as temp_file:
                temp_file.write(weights_file.read())
                temp_path = Path(temp_file.name)

    try:
        with h5py.File(temp_path, "r") as h5:
            # The .keras archive stores these as HDF5 names containing
            # backslashes rather than as nested groups.

            conv1d = h5[
                r"_layer_checkpoint_dependencies\conv1d"
            ]["vars"]

            lstm_cell = h5[
                r"_layer_checkpoint_dependencies\lstm\cell"
            ]["vars"]

            attention_score = h5[
                r"_layer_checkpoint_dependencies\attention\score"
            ]["vars"]

            dense = h5[
                r"_layer_checkpoint_dependencies\dense"
            ]["vars"]

            dense_2 = h5[
                r"_layer_checkpoint_dependencies\dense_2"
            ]["vars"]

            model.get_layer("conv1d_32").set_weights(
                [
                    np.asarray(conv1d["0"]),
                    np.asarray(conv1d["1"]),
                ]
            )

            model.get_layer("lstm_64").set_weights(
                [
                    np.asarray(lstm_cell["0"]),
                    np.asarray(lstm_cell["1"]),
                    np.asarray(lstm_cell["2"]),
                ]
            )

            model.get_layer("attention_layer").score.set_weights(
                [
                    np.asarray(attention_score["0"]),
                    np.asarray(attention_score["1"]),
                ]
            )

            model.get_layer("dense_32").set_weights(
                [
                    np.asarray(dense["0"]),
                    np.asarray(dense["1"]),
                ]
            )

            model.get_layer("dense_softmax").set_weights(
                [
                    np.asarray(dense_2["0"]),
                    np.asarray(dense_2["1"]),
                ]
            )

    finally:
        temp_path.unlink(missing_ok=True)


class CNNLSTMAttention:
    """Public inference interface for the final temporal IDS model."""

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata not found: {METADATA_PATH}"
            )

        if not SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Scaler not found: {SCALER_PATH}"
            )

        self.metadata = json.loads(
            METADATA_PATH.read_text(encoding="utf-8")
        )

        self.labels = [
            label
            for label, index in sorted(
                self.metadata["label_mapping"].items(),
                key=lambda item: item[1],
            )
        ]

        scaler = json.loads(
            SCALER_PATH.read_text(encoding="utf-8")
        )

        self.mean = np.asarray(
            scaler["mean"],
            dtype=np.float32,
        )

        self.std = np.asarray(
            scaler["std"],
            dtype=np.float32,
        )

        self.feature_order = list(
            scaler["feature_order"]
        )

        self.input_shape = (10, 5)

        self.model = build_model()

        load_weights_from_keras_archive(
            self.model,
            MODEL_PATH,
        )


    def predict(self, sequence):
        """Predict one temporal sequence.

        Parameters
        ----------
        sequence:
            Array-like object with shape (10, 5).

        Returns
        -------
        dict
            Prediction, confidence and class probabilities.
        """

        values = np.asarray(
            sequence,
            dtype=np.float32,
        )

        if values.shape != self.input_shape:
            raise ValueError(
                "Expected sequence shape "
                f"{self.input_shape}, got {values.shape}"
            )

        values = (
            values - self.mean
        ) / self.std

        probabilities = self.model.predict(
            values[None, ...],
            verbose=0,
        )[0]

        index = int(
            np.argmax(probabilities)
        )

        return {
            "prediction": self.labels[index],
            "confidence": float(
                probabilities[index]
            ),
            "probabilities": {
                label: float(
                    probabilities[i]
                )
                for i, label in enumerate(self.labels)
            },
        }