"""Inference for the trained CNN + LSTM + Attention model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "temporal_sequences"
META = ROOT / "data" / "metadata"


class CNNLSTMAttention:
    def __init__(self):
        self.metadata = json.loads((META / "temporal_sequence_metadata.json").read_text(encoding="utf-8"))
        self.labels = list(self.metadata["label_mapping"])
        scaler = json.loads((DATA / "scaler.json").read_text(encoding="utf-8"))
        self.mean = np.asarray(scaler["mean"], dtype=np.float32)
        self.scale = np.asarray(scaler["scale"], dtype=np.float32)
        self.model = tf.keras.models.load_model(ROOT / "models" / "trained" / "cnn_lstm_attention.keras", custom_objects={"Attention": __import__("models.train_cnn_lstm_attention", fromlist=["Attention"]).Attention})

    def predict(self, sequence):
        values = np.asarray(sequence, dtype=np.float32)
        if values.shape != (10, 5):
            raise ValueError("sequence must have shape (10, 5)")
        if not np.isfinite(values).all():
            raise ValueError("sequence values must be finite")
        values = (values - self.mean) / self.scale
        probabilities = self.model.predict(values[None, ...], verbose=0)[0]
        index = int(np.argmax(probabilities))
        return {"prediction": self.labels[index], "confidence": float(probabilities[index]), "probabilities": {label: float(probabilities[i]) for i, label in enumerate(self.labels)}}
