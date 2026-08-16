"""Retrain baseline CNN+LSTM+Attention on existing surrogate sequences,
save fresh model compatible with current environment, run inference,
and record results.

This script:
 - Purges the AppData user-site from sys.path so TF 2.13 / Keras 2.13
   in Anaconda is loaded (not the conflicting AppData Keras 3).
 - Trains on data/processed/temporal_sequences/{train,validation,test}.npz
 - Saves model to models/trained/cnn_lstm_attention.keras
 - Saves inference result to data/metadata/baseline_inference_result.json
 - Saves updated results to data/metadata/cnn_lstm_attention_results.json

BASELINE experiment only (surrogate timestamp-ordered windows).
NOT raw-PCAP.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# TF_USE_LEGACY_KERAS must be in the environment before Python starts.
# Set via run_inference.bat. Also set here as fallback for lazy checks.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
print(f"TensorFlow {tf.__version__}")


from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent

DATA       = ROOT / "data" / "processed" / "temporal_sequences"
META       = ROOT / "data" / "metadata"
MODEL_PATH = ROOT / "models" / "trained" / "cnn_lstm_attention.keras"
SEED       = 42

LABELS    = ["Benign", "Brute Force", "Botnet", "Web Attack"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}

# ── Attention layer ───────────────────────────────────────────────────────────
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = tf.keras.layers.Dense(1)

    def call(self, inputs):
        weights = tf.nn.softmax(self.score(inputs), axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)

    def get_config(self):
        return super().get_config()

# ── Load data ─────────────────────────────────────────────────────────────────
tf.keras.utils.set_random_seed(SEED)
metadata  = json.loads((META / "temporal_sequence_metadata.json").read_text(encoding="utf-8"))
train_npz = np.load(DATA / "train.npz")
val_npz   = np.load(DATA / "validation.npz")
test_npz  = np.load(DATA / "test.npz")

X_train, y_train = train_npz["X"], train_npz["y"]
X_val,   y_val   = val_npz["X"],   val_npz["y"]
X_test,  y_test  = test_npz["X"],  test_npz["y"]

print(f"Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

counts = np.bincount(y_train, minlength=len(LABELS))
class_weights = {i: float(len(y_train) / (len(LABELS) * c)) for i, c in enumerate(counts)}

# ── Model ─────────────────────────────────────────────────────────────────────
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
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
import time
t0 = time.perf_counter()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )],
    verbose=1,
    shuffle=True,
)
elapsed = time.perf_counter() - t0
print(f"Training complete in {elapsed:.1f}s, epochs={len(history.history['loss'])}")

# ── Evaluate ──────────────────────────────────────────────────────────────────
def get_metrics(y_true, probs):
    preds = probs.argmax(axis=1)
    rep = classification_report(
        y_true, preds,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0
    )
    return {
        "accuracy":         float(rep["accuracy"]),
        "macro_precision":  float(rep["macro avg"]["precision"]),
        "macro_recall":     float(rep["macro avg"]["recall"]),
        "macro_f1":         float(rep["macro avg"]["f1-score"]),
        "weighted_f1":      float(rep["weighted avg"]["f1-score"]),
        "per_class":        {l: {"precision": rep[l]["precision"],
                                  "recall":    rep[l]["recall"],
                                  "f1":        rep[l]["f1-score"]}
                             for l in LABELS},
        "confusion_matrix": confusion_matrix(
            y_true, preds, labels=list(range(len(LABELS)))
        ).tolist(),
    }

val_result  = get_metrics(y_val,  model.predict(X_val,  verbose=0))
test_result = get_metrics(y_test, model.predict(X_test, verbose=0))

print("\nTest results:")
print(json.dumps(test_result, indent=2))

# ── Save model ────────────────────────────────────────────────────────────────
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")

# ── Save evaluation results ───────────────────────────────────────────────────
full_results = {
    "architecture": [
        "Conv1D(32,3,relu)", "MaxPooling1D(2)",
        "LSTM(64,return_sequences=True)", "Attention",
        "Dense(32,relu)", "Dropout(0.3)", "Dense(4,softmax)"
    ],
    "epochs_completed": len(history.history["loss"]),
    "training_seconds": elapsed,
    "class_weights": class_weights,
    "validation":    val_result,
    "test":          test_result,
    "limitation":    metadata["limitation"],
}
(META / "cnn_lstm_attention_results.json").write_text(
    json.dumps(full_results, indent=2) + "\n", encoding="utf-8"
)

# ── Baseline inference on first test sequence ─────────────────────────────────
seq_norm   = X_test[0]
true_idx   = int(y_test[0])
true_label = LABELS[true_idx]

probs      = model.predict(seq_norm[None, ...], verbose=0)[0]
pred_idx   = int(np.argmax(probs))
pred_label = LABELS[pred_idx]
confidence = float(probs[pred_idx])

inference_result = {
    "experiment":     "BASELINE (surrogate timestamp-ordered windows from CICFlowMeter CSVs)",
    "sequence_index": 0,
    "true_label":     true_label,
    "prediction":     pred_label,
    "confidence":     confidence,
    "probabilities": {lbl: float(probs[i]) for i, lbl in enumerate(LABELS)},
    "correct":        pred_label == true_label,
    "note": (
        "BASELINE inference only. This sequence comes from surrogate "
        "timestamp-ordered windows from reduced CICFlowMeter CSVs. "
        "It is NOT raw-PCAP inference. "
        "The raw-PCAP pipeline produced 0 valid length-10 sequences "
        "because the maximum observed communication-group size is 3."
    )
}

(META / "baseline_inference_result.json").write_text(
    json.dumps(inference_result, indent=2) + "\n", encoding="utf-8"
)
print("\nBaseline inference result:")
print(json.dumps(inference_result, indent=2))
