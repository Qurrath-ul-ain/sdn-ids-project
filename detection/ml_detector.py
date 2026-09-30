"""CNN-LSTM-Attention ML detector for the SDN-IDS project."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import tensorflow as tf

from inference.cnn_lstm_attention import CNNLSTMAttention


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_cicflow_temporal"
    / "cnn_lstm_attention_final.keras"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_cicflow_temporal"
    / "scaler.json"
)

SEQUENCE_LENGTH = 10

FEATURE_NAMES = [
    "destination_port",
    "protocol",
    "packet_count",
    "byte_count",
    "flow_duration_us",
]

CLASS_NAMES = [
    "Benign",
    "Brute Force",
    "Botnet",
    "Web Attack",
]


class Attention(tf.keras.layers.Layer):
    """Exact Attention layer used when the model was trained."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = tf.keras.layers.Dense(1)

    def call(self, inputs):
        weights = tf.nn.softmax(self.score(inputs), axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)

    def get_config(self):
        config = super().get_config()
        return config


class MLDetector:
    """Maintain 10-flow windows and classify them with CNN-LSTM-Attention."""

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

        scaler = json.loads(SCALER_PATH.read_text(encoding="utf-8"))

        if scaler.get("feature_order") != FEATURE_NAMES:
            raise ValueError(
                "Scaler feature order does not match the trained model."
            )

        self.mean = np.asarray(scaler["mean"], dtype=np.float32)
        self.std = np.asarray(scaler["std"], dtype=np.float32)

        if len(self.mean) != len(FEATURE_NAMES):
            raise ValueError("Scaler mean dimension mismatch.")

        if len(self.std) != len(FEATURE_NAMES):
            raise ValueError("Scaler std dimension mismatch.")

        if np.any(self.std == 0):
            raise ValueError("Scaler contains zero standard deviation.")

        self.inference = CNNLSTMAttention()
        self.model = self.inference.model

        if self.model.count_params() != 27621:
            raise ValueError(
                f"Unexpected model parameter count: "
                f"{self.model.count_params()} (expected 27621)"
            )

        if self.model.output_shape[-1] != len(CLASS_NAMES):
            raise ValueError("Model output count does not match class names.")

        # Each conversation gets a temporal window of DISTINCT flows.
        # A flow may appear in multiple OpenFlow statistics polls, so
        # duplicate observations of the same flow are not appended.
        self.windows = defaultdict(
            lambda: deque(maxlen=SEQUENCE_LENGTH)
        )

    @staticmethod
    def _flow_key(flow: dict) -> tuple:
        """Keep temporal flows grouped by source/destination conversation."""

        return (
            flow.get("src_ip"),
            flow.get("dst_ip"),
            flow.get("dst_port"),
            flow.get("protocol"),
        )

    @staticmethod
    def _feature_values(flow: dict) -> list[float]:
        """Extract the exact five features used during training."""

        values = [
            flow.get("dst_port", 0),
            flow.get("protocol", 0),
            flow.get("packet_count", 0),
            flow.get("byte_count", 0),
            flow.get("flow_duration_us", 0),
        ]

        result = []

        for name, value in zip(FEATURE_NAMES, values):
            try:
                value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Feature '{name}' must be numeric."
                ) from error

            if not np.isfinite(value):
                raise ValueError(
                    f"Feature '{name}' must be finite."
                )

            result.append(value)

        return result

    def add_flow(self, flow: dict) -> dict | None:
        """
        Add one distinct controller flow to its temporal window.

        OpenFlow statistics are polled repeatedly, so the same flow can
        appear in multiple replies. A complete five-tuple identifies one
        flow and prevents repeated polling observations from becoming
        duplicate temporal samples.

        Returns None until 10 distinct flows are available.
        Once 10 distinct flows are available, a new flow triggers
        an ML prediction.
        """

        key = self._flow_key(flow)
        features = self._feature_values(flow)

        flow_id = (
            flow.get("src_ip"),
            flow.get("dst_ip"),
            flow.get("src_port"),
            flow.get("dst_port"),
            flow.get("protocol"),
        )

        window = self.windows[key]

        # If this exact flow was already observed, update its latest
        # feature values but do not add a duplicate temporal sample.
        for index, (existing_id, _) in enumerate(window):
            if existing_id == flow_id:
                window[index] = (existing_id, features)
                return None

        # This is a new flow in the conversation.
        window.append((flow_id, features))

        if len(window) < SEQUENCE_LENGTH:
            return None

        sequence = np.asarray(
            [item[1] for item in window],
            dtype=np.float32,
        )

        print("\n========== ACTUAL ML SEQUENCE ==========")
        for i, row in enumerate(sequence, 1):
            print(f"Flow {i:02d}: {row.tolist()}")
        print("=========================================")

        result = self.inference.predict(sequence)

        print("========== ML RESULT ====================")
        print(f"Prediction : {result['prediction']}")
        print(f"Confidence : {result['confidence'] * 100:.4f}%")
        for label, probability in result["probabilities"].items():
            print(f"{label:12s}: {probability * 100:.4f}%")
        print("=========================================")

        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "sequence_length": SEQUENCE_LENGTH,
            "features": FEATURE_NAMES,
        }
