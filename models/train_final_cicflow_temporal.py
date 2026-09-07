"""Train the final CNN + LSTM + Attention classifier on the leakage-controlled temporal dataset.

Evaluation Protocol:
- Class-stratified temporal evaluation protocol
- Uses strictly data/processed/final_cicflow_temporal/
- Validates all shapes, class counts, and architecture constraints
- Test set evaluated exactly once after training
- Saves all artifacts into models/final_cicflow_temporal/
- Baseline files remain completely untouched
"""

from __future__ import annotations

import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "final_cicflow_temporal"
OUT_DIR = ROOT / "models" / "final_cicflow_temporal"

SEED = 42
SEQUENCE_LENGTH = 10
NUM_FEATURES = 5
NUM_CLASSES = 4
BATCH_SIZE = 64
MAX_EPOCHS = 20
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
PATIENCE = 3

FEATURE_NAMES = [
    "destination_port",
    "protocol",
    "packet_count",
    "byte_count",
    "flow_duration_us",
]
EXPECTED_CLASSES = ["Benign", "Brute Force", "Botnet", "Web Attack"]
EXPECTED_COUNTS = {
    "train": {"Benign": 640, "Brute Force": 640, "Botnet": 640, "Web Attack": 640},
    "validation": {"Benign": 130, "Brute Force": 130, "Botnet": 130, "Web Attack": 130},
    "test": {"Benign": 113, "Brute Force": 113, "Botnet": 113, "Web Attack": 113},
}
EXPECTED_SHAPES = {
    "train": ((2560, 10, 5), (2560,)),
    "validation": ((520, 10, 5), (520,)),
    "test": ((452, 10, 5), (452,)),
}
EXPECTED_PARAMS = 27621


class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = tf.keras.layers.Dense(1)

    def call(self, inputs):
        weights = tf.nn.softmax(self.score(inputs), axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)

    def get_config(self):
        config = super().get_config()
        return config


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray, labels: list[str]) -> dict:
    predictions = probabilities.argmax(axis=1)
    report = classification_report(
        y_true,
        predictions,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, predictions, labels=list(range(len(labels))))
    
    per_class = {}
    for idx, label in enumerate(labels):
        per_class[label] = {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1_score": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        
    correct = int((predictions == y_true).sum())
    incorrect = int((predictions != y_true).sum())
    
    return {
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "total_evaluated": len(y_true),
        "correct_predictions": correct,
        "incorrect_predictions": incorrect,
    }


def main():
    print("=" * 70)
    print("TRAINING FINAL CNN-LSTM-ATTENTION (LEAKAGE-CONTROLLED MAIN EXPERIMENT)")
    print("=" * 70)

    # 1. Environment & Determinism Setup
    tf.keras.utils.set_random_seed(SEED)

    env_info = {
        "python_version": sys.version,
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "random_seed": SEED,
    }
    print(f"Environment: Python {sys.version.split()[0]} | TensorFlow {tf.__version__} | NumPy {np.__version__}")

    # 2. Load & Validate Dataset Partitions
    print("\nStep 1: Loading and verifying dataset partitions...")
    train_npz = np.load(DATA_DIR / "train.npz")
    val_npz = np.load(DATA_DIR / "validation.npz")
    test_npz = np.load(DATA_DIR / "test.npz")

    X_train, y_train = train_npz["X"], train_npz["y"]
    X_val, y_val = val_npz["X"], val_npz["y"]
    X_test, y_test = test_npz["X"], test_npz["y"]

    # Verify shapes
    assert X_train.shape == EXPECTED_SHAPES["train"][0] and y_train.shape == EXPECTED_SHAPES["train"][1], f"Train shape mismatch: {X_train.shape}, {y_train.shape}"
    assert X_val.shape == EXPECTED_SHAPES["validation"][0] and y_val.shape == EXPECTED_SHAPES["validation"][1], f"Validation shape mismatch: {X_val.shape}, {y_val.shape}"
    assert X_test.shape == EXPECTED_SHAPES["test"][0] and y_test.shape == EXPECTED_SHAPES["test"][1], f"Test shape mismatch: {X_test.shape}, {y_test.shape}"

    # Verify class counts
    for split_name, y_data in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        counts = np.bincount(y_data, minlength=NUM_CLASSES)
        for idx, label in enumerate(EXPECTED_CLASSES):
            expected = EXPECTED_COUNTS[split_name][label]
            actual = int(counts[idx])
            assert actual == expected, f"{split_name} count mismatch for {label}: expected {expected}, got {actual}"

    print("  Train shape      :", X_train.shape, "Classes:", dict(zip(EXPECTED_CLASSES, np.bincount(y_train))))
    print("  Validation shape :", X_val.shape, "Classes:", dict(zip(EXPECTED_CLASSES, np.bincount(y_val))))
    print("  Test shape       :", X_test.shape, "Classes:", dict(zip(EXPECTED_CLASSES, np.bincount(y_test))))

    # Verify Scaler Metadata
    scaler_path = DATA_DIR / "scaler.json"
    assert scaler_path.exists(), f"Scaler not found at {scaler_path}"
    scaler_data = json.loads(scaler_path.read_text(encoding="utf-8"))
    assert scaler_data.get("scaler_fit_partition") == "train", "Scaler was not fit strictly on training data!"
    print("  Scaler check     : PASSED (fitted strictly on training partition)")

    # 3. Model Construction & Architecture Verification
    print("\nStep 2: Building model architecture...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES), name="input_temporal_window"),
        tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu", name="conv1d_32"),
        tf.keras.layers.MaxPooling1D(2, name="max_pooling1d"),
        tf.keras.layers.LSTM(64, return_sequences=True, name="lstm_64"),
        Attention(name="attention_layer"),
        tf.keras.layers.Dense(32, activation="relu", name="dense_32"),
        tf.keras.layers.Dropout(DROPOUT_RATE, name="dropout_03"),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="dense_softmax"),
    ], name="CNN_LSTM_Attention_Final")

    total_params = model.count_params()
    assert total_params == EXPECTED_PARAMS, f"Parameter count mismatch: {total_params} != {EXPECTED_PARAMS}"
    print(f"  Architecture     : Conv1D(32) -> MaxPool1D -> LSTM(64) -> Attention -> Dense(32) -> Dropout(0.3) -> Softmax(4)")
    print(f"  Total Parameters : {total_params:,} (matches baseline exactly: {EXPECTED_PARAMS:,})")

    # Capture model summary
    summary_io = io.StringIO()
    model.summary(print_fn=lambda x: summary_io.write(x + "\n"))
    model_summary_str = summary_io.getvalue()

    # 4. Compilation & Training
    print("\nStep 3: Compiling and training model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    t0 = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1,
        shuffle=True,
    )
    training_time_sec = time.perf_counter() - t0
    epochs_completed = len(history.history["loss"])
    print(f"\n  Training completed in {training_time_sec:.2f}s ({epochs_completed} epochs)")

    # 5. Validation Evaluation
    print("\nStep 4: Evaluating best model on validation set...")
    val_probs = model.predict(X_val, verbose=0)
    val_metrics = compute_metrics(y_val, val_probs, EXPECTED_CLASSES)
    print(f"  Validation Accuracy : {val_metrics['accuracy'] * 100:.2f}%")
    print(f"  Validation Macro F1 : {val_metrics['macro_f1'] * 100:.2f}%")

    # 6. Final Test Evaluation (Executed Exactly Once)
    print("\nStep 5: Executing final held-out test evaluation...")
    test_probs = model.predict(X_test, verbose=0)
    
    # Sanity checks on predictions
    assert not np.isnan(test_probs).any(), "NaN detected in test predictions!"
    assert not np.isinf(test_probs).any(), "Inf detected in test predictions!"
    test_preds = test_probs.argmax(axis=1)
    assert set(np.unique(test_preds)).issubset({0, 1, 2, 3}), f"Invalid class IDs: {np.unique(test_preds)}"
    
    test_metrics = compute_metrics(y_test, test_probs, EXPECTED_CLASSES)
    
    # Assert sanity on evaluation
    assert test_metrics["total_evaluated"] == 452, f"Evaluated count mismatch: {test_metrics['total_evaluated']}"
    assert sum(sum(row) for row in test_metrics["confusion_matrix"]) == 452, "Confusion matrix sum mismatch!"

    print("\n" + "=" * 70)
    print(f"FINAL HELD-OUT TEST RESULTS:")
    print(f"  Test Accuracy      : {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro Precision    : {test_metrics['macro_precision'] * 100:.2f}%")
    print(f"  Macro Recall       : {test_metrics['macro_recall'] * 100:.2f}%")
    print(f"  Macro F1-Score     : {test_metrics['macro_f1'] * 100:.2f}%")
    print(f"  Weighted F1-Score  : {test_metrics['weighted_f1'] * 100:.2f}%")
    print(f"  Correct / Total    : {test_metrics['correct_predictions']} / {test_metrics['total_evaluated']} ({test_metrics['incorrect_predictions']} incorrect)")
    print("=" * 70)

    print("\nPer-Class Performance:")
    for label in EXPECTED_CLASSES:
        pcm = test_metrics["per_class"][label]
        print(f"  {label:<12}: Precision = {pcm['precision']*100:>6.2f}% | Recall = {pcm['recall']*100:>6.2f}% | F1 = {pcm['f1_score']*100:>6.2f}% | Support = {pcm['support']}")

    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
    print(f"Labels: {EXPECTED_CLASSES}")
    cm_arr = np.array(test_metrics["confusion_matrix"])
    print(cm_arr)

    # 7. Save Artifacts Separately
    print("\nStep 6: Saving final results and artifacts...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Trained Model
    model_save_path = OUT_DIR / "cnn_lstm_attention_final.keras"
    model.save(model_save_path)

    # 2. Model Summary Text
    (OUT_DIR / "model_summary.txt").write_text(model_summary_str, encoding="utf-8")

    # 3. Training Config
    training_config = {
        "environment": env_info,
        "model_name": "CNN_LSTM_Attention_Final",
        "architecture_summary": [
            "Input(shape=(10, 5))",
            "Conv1D(32, 3, padding='same', activation='relu')",
            "MaxPooling1D(2)",
            "LSTM(64, return_sequences=True)",
            "Attention(Dense(1))",
            "Dense(32, activation='relu')",
            "Dropout(0.3)",
            "Dense(4, activation='softmax')",
        ],
        "total_parameters": total_params,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "sparse_categorical_crossentropy",
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "epochs_completed": epochs_completed,
        "early_stopping": {
            "monitor": "val_loss",
            "patience": PATIENCE,
            "restore_best_weights": True,
        },
        "training_time_seconds": round(training_time_sec, 2),
    }
    (OUT_DIR / "training_config.json").write_text(json.dumps(training_config, indent=2) + "\n", encoding="utf-8")

    # 4. Training History
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (OUT_DIR / "training_history.json").write_text(json.dumps(history_dict, indent=2) + "\n", encoding="utf-8")

    # 5. Confusion Matrix JSON
    cm_dict = {
        "classes": EXPECTED_CLASSES,
        "matrix": test_metrics["confusion_matrix"],
        "total_support": 452,
    }
    (OUT_DIR / "confusion_matrix.json").write_text(json.dumps(cm_dict, indent=2) + "\n", encoding="utf-8")

    # 6. Final Results JSON (as specified in requirement)
    final_results = {
        "experiment": "final_cicflow_temporal",
        "dataset": "CICFlowMeter",
        "evaluation_protocol": "class-stratified temporal evaluation",
        "protocol_disclaimer": "This is a class-stratified temporal evaluation protocol; it is NOT a global chronological deployment stream simulation.",
        "terminology_note": "Sequences are chronological temporal flow windows constructed within isolated class partitions. They are not reconstructed network sessions.",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sequence_length": SEQUENCE_LENGTH,
        "stride": 1,
        "features": FEATURE_NAMES,
        "input_shape": [SEQUENCE_LENGTH, NUM_FEATURES],
        "classes": EXPECTED_CLASSES,
        "train_sequences": len(X_train),
        "validation_sequences": len(X_val),
        "test_sequences": len(X_test),
        "test_accuracy": test_metrics["accuracy"],
        "macro_precision": test_metrics["macro_precision"],
        "macro_recall": test_metrics["macro_recall"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_precision": test_metrics["weighted_precision"],
        "weighted_recall": test_metrics["weighted_recall"],
        "weighted_f1": test_metrics["weighted_f1"],
        "per_class": test_metrics["per_class"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "baseline_accuracy": 0.9744,
        "accuracy_comparison": {
            "original_baseline": 0.9744,
            "final_leakage_controlled": test_metrics["accuracy"],
            "difference": round(test_metrics["accuracy"] - 0.9744, 4),
            "note": "Final model evaluated under strict class-stratified temporal boundaries with 9-flow quarantine and zero partition leakage.",
        },
        "validation_metrics": val_metrics,
        "training_configuration": training_config,
    }
    (OUT_DIR / "final_results.json").write_text(json.dumps(final_results, indent=2) + "\n", encoding="utf-8")

    print(f"\nAll final artifacts successfully written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
