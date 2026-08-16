"""Train the mandatory CNN + LSTM + Attention temporal classifier."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "temporal_sequences"
META = ROOT / "data" / "metadata"
MODEL_PATH = ROOT / "models" / "trained" / "cnn_lstm_attention.keras"
RESULT_PATH = META / "cnn_lstm_attention_results.json"
SEED = 42


class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = tf.keras.layers.Dense(1)

    def call(self, inputs):
        weights = tf.nn.softmax(self.score(inputs), axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)


def metrics(y_true, probabilities, labels):
    predictions = probabilities.argmax(axis=1)
    report = classification_report(y_true, predictions, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {label: {"precision": report[label]["precision"], "recall": report[label]["recall"], "f1": report[label]["f1-score"]} for label in labels},
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=list(range(len(labels)))).tolist(),
    }


def main():
    tf.keras.utils.set_random_seed(SEED)
    metadata = json.loads((META / "temporal_sequence_metadata.json").read_text(encoding="utf-8"))
    labels = list(metadata["label_mapping"])
    train = np.load(DATA / "train.npz")
    validation = np.load(DATA / "validation.npz")
    test = np.load(DATA / "test.npz")
    counts = np.bincount(train["y"], minlength=len(labels))
    class_weights = {index: float(len(train["y"]) / (len(labels) * count)) for index, count in enumerate(counts)}
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10, 5)),
        tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.LSTM(64, return_sequences=True),
        Attention(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    started = time.perf_counter()
    history = model.fit(train["X"], train["y"], validation_data=(validation["X"], validation["y"]), epochs=20, batch_size=64, class_weight=class_weights, callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)], verbose=0, shuffle=True)
    elapsed = time.perf_counter() - started
    validation_result = metrics(validation["y"], model.predict(validation["X"], verbose=0), labels)
    test_result = metrics(test["y"], model.predict(test["X"], verbose=0), labels)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    result = {"architecture": ["Conv1D(32,3,relu)", "MaxPooling1D(2)", "LSTM(64,return_sequences=True)", "Attention", "Dense(32,relu)", "Dropout(0.3)", "Dense(4,softmax)"], "epochs_completed": len(history.history["loss"]), "training_seconds": elapsed, "class_weights": class_weights, "validation": validation_result, "test": test_result, "limitation": metadata["limitation"]}
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
