"""Inference interface for the trained SDN-IDS models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
MODEL_DIR = PROJECT_ROOT / "models" / "trained"
PREPROCESSING_DIR = PROJECT_ROOT / "models" / "preprocessing"
ModelMode = Literal["hybrid", "random_forest"]


class HybridIDS:
    """Load trained artifacts and predict one canonical flow dictionary."""

    def __init__(self) -> None:
        self.feature_names = self._read_json(METADATA_DIR / "feature_list.json")
        self.label_mapping = self._read_json(METADATA_DIR / "label_mapping.json")
        self.scaler = self._read_json(PREPROCESSING_DIR / "scaler.json")
        self.hybrid_metadata = self._read_json(METADATA_DIR / "hybrid_metadata.json")
        self.rf_metadata = self._read_json(METADATA_DIR / "random_forest_metadata.json")
        self.dnn_metadata = self._read_json(METADATA_DIR / "dnn_metadata.json")

        self.class_names = list(self.label_mapping)
        self._validate_metadata()
        self.rf = joblib.load(MODEL_DIR / "random_forest.joblib")
        self.dnn = tf.keras.models.load_model(MODEL_DIR / "dnn.keras")
        self.rf_class_order = [str(value) for value in self.rf.classes_]
        self._validate_model_classes()

    @staticmethod
    def _read_json(path: Path) -> object:
        with path.open(encoding="utf-8") as input_file:
            return json.load(input_file)

    def _validate_metadata(self) -> None:
        if self.feature_names != self.scaler["feature_order"]:
            raise ValueError("Feature metadata and scaler feature order do not match")
        if self.feature_names != self.rf_metadata["feature_order"]:
            raise ValueError("Feature metadata and Random Forest metadata feature order do not match")
        if self.feature_names != self.dnn_metadata["feature_order"]:
            raise ValueError("Feature metadata and DNN metadata feature order do not match")
        if self.feature_names != self.hybrid_metadata["feature_order"]:
            raise ValueError("Feature metadata and hybrid metadata feature order do not match")
        if self.rf_metadata["label_mapping"] != self.label_mapping:
            raise ValueError("Random Forest label metadata does not match label metadata")
        if self.dnn_metadata["label_mapping"] != self.label_mapping:
            raise ValueError("DNN label metadata does not match label metadata")
        if self.hybrid_metadata["class_order"] != self.class_names:
            raise ValueError("Hybrid class order does not match label metadata")
        if self.hybrid_metadata["selected_weights"] != {"rf_weight": 0.6, "dnn_weight": 0.4}:
            raise ValueError("Hybrid metadata does not contain the final 0.6/0.4 weights")

    def _validate_model_classes(self) -> None:
        if set(self.rf_class_order) != set(self.class_names):
            raise ValueError("Random Forest classes do not match label metadata")
        if self.dnn.output_shape[-1] != len(self.class_names):
            raise ValueError("DNN output count does not match label metadata")

    def _prepare_features(self, flow: dict[str, object]) -> pd.DataFrame:
        if not isinstance(flow, dict):
            raise TypeError("flow must be a dictionary")
        received = set(flow)
        required = set(self.feature_names)
        missing = [name for name in self.feature_names if name not in received]
        unexpected = sorted(received - required)
        if missing:
            raise ValueError(f"Missing required feature(s): {', '.join(missing)}")
        if unexpected:
            raise ValueError(f"Unexpected feature(s): {', '.join(unexpected)}")

        values: list[float] = []
        for name in self.feature_names:
            try:
                value = float(flow[name])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Feature '{name}' must be numeric") from error
            if not np.isfinite(value):
                raise ValueError(f"Feature '{name}' must be finite")
            values.append(value)

        raw = np.asarray([values], dtype=np.float64)
        mean = np.asarray(self.scaler["mean"], dtype=np.float64)
        scale = np.asarray(self.scaler["scale"], dtype=np.float64)
        if len(mean) != len(self.feature_names) or len(scale) != len(self.feature_names):
            raise ValueError("Scaler dimensions do not match feature metadata")
        transformed = (raw - mean) / scale
        return pd.DataFrame(transformed, columns=self.feature_names)

    def _rf_probabilities(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = self.rf.predict_proba(features)[0]
        indexes = [self.rf_class_order.index(name) for name in self.class_names]
        return np.asarray(probabilities[indexes], dtype=np.float64)

    def _dnn_probabilities(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = self.dnn.predict(features.to_numpy(dtype=np.float32), verbose=0)[0]
        result = np.asarray(probabilities, dtype=np.float64)
        if result.shape != (len(self.class_names),):
            raise ValueError("DNN probability output does not match class metadata")
        return result

    def predict(self, flow: dict[str, object], model: ModelMode = "hybrid") -> dict[str, object]:
        """Predict a flow using the final hybrid or RF-only model."""
        if model not in ("hybrid", "random_forest"):
            raise ValueError("model must be 'hybrid' or 'random_forest'")
        features = self._prepare_features(flow)
        rf_probabilities = self._rf_probabilities(features)
        if model == "random_forest":
            probabilities = rf_probabilities
        else:
            dnn_probabilities = self._dnn_probabilities(features)
            probabilities = 0.6 * rf_probabilities + 0.4 * dnn_probabilities
        probabilities = probabilities / probabilities.sum()
        prediction_index = int(np.argmax(probabilities))
        return {
            "prediction": self.class_names[prediction_index],
            "confidence": float(probabilities[prediction_index]),
            "probabilities": {
                name: float(probabilities[index])
                for index, name in enumerate(self.class_names)
            },
        }
