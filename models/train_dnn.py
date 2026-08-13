"""Train and evaluate the four-class tabular SDN-IDS DNN."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
MODEL_DIR = PROJECT_ROOT / "models" / "trained"
FEATURE_LIST_PATH = METADATA_DIR / "feature_list.json"
LABEL_MAPPING_PATH = METADATA_DIR / "label_mapping.json"
MODEL_PATH = MODEL_DIR / "dnn.keras"
METADATA_PATH = METADATA_DIR / "dnn_metadata.json"
RANDOM_SEED = 42
BATCH_SIZE = 8192
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 4


def read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2)
        output_file.write("\n")


def load_split(split_name: str, feature_names: list[str], label_mapping: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    required_columns = [*feature_names, "Label"]
    frame = pd.read_csv(SPLITS_DIR / f"{split_name}.csv", usecols=required_columns, low_memory=False)
    if frame.columns.tolist() != required_columns:
        raise ValueError(f"{split_name} split does not preserve the canonical feature order")
    labels = frame.pop("Label").map(label_mapping)
    if labels.isna().any():
        raise ValueError(f"{split_name} split contains an unknown label")
    return frame.to_numpy(dtype=np.float32), labels.to_numpy(dtype=np.int32)


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray, labels: list[str]) -> dict[str, object]:
    predictions = probabilities.argmax(axis=1)
    numeric_labels = list(range(len(labels)))
    report = classification_report(
        y_true,
        predictions,
        labels=numeric_labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=numeric_labels).tolist(),
    }


def main() -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    feature_names = read_json(FEATURE_LIST_PATH)
    label_mapping = read_json(LABEL_MAPPING_PATH)
    if not isinstance(feature_names, list) or not isinstance(label_mapping, dict):
        raise ValueError("Feature and label metadata have invalid formats")
    labels = list(label_mapping)

    train_features, train_labels = load_split("train", feature_names, label_mapping)
    validation_features, validation_labels = load_split("validation", feature_names, label_mapping)

    train_class_counts = np.bincount(train_labels, minlength=len(labels))
    total_train = len(train_labels)
    class_weights = {
        class_index: float(total_train / (len(labels) * count))
        for class_index, count in enumerate(train_class_counts)
    }

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(len(feature_names),), name="flow_features"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(len(labels), activation="softmax"),
        ],
        name="sdn_ids_dnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        mode="min",
    )

    training_started = time.perf_counter()
    history = model.fit(
        train_features,
        train_labels,
        validation_data=(validation_features, validation_labels),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=2,
        shuffle=True,
    )
    training_seconds = time.perf_counter() - training_started

    validation_probabilities = model.predict(validation_features, batch_size=BATCH_SIZE, verbose=0)
    validation_metrics = calculate_metrics(validation_labels, validation_probabilities, labels)

    # Load and evaluate the held-out test set only after training/model selection.
    test_features, test_labels = load_split("test", feature_names, label_mapping)
    test_probabilities = model.predict(test_features, batch_size=BATCH_SIZE, verbose=0)
    test_metrics = calculate_metrics(test_labels, test_probabilities, labels)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    metadata = {
        "model_type": "tabular DNN",
        "tensorflow_version": tf.__version__,
        "architecture": [
            "Dense(128, relu)",
            "BatchNormalization",
            "Dropout(0.30)",
            "Dense(64, relu)",
            "BatchNormalization",
            "Dropout(0.30)",
            "Dense(32, relu)",
            "Dense(4, softmax)",
        ],
        "feature_order": feature_names,
        "label_mapping": label_mapping,
        "configuration": {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "loss": "sparse_categorical_crossentropy",
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "epochs_completed": len(history.history["loss"]),
            "early_stopping_monitor": "val_loss",
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "restore_best_weights": True,
            "random_seed": RANDOM_SEED,
            "class_weights": {labels[index]: weight for index, weight in class_weights.items()},
        },
        "training_rows": len(train_features),
        "validation_rows": len(validation_features),
        "test_rows": len(test_features),
        "training_seconds": training_seconds,
        "best_validation_loss": float(min(history.history["val_loss"])),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "preprocessing_source": "Existing prepared splits using the training-fitted scaler",
        "controller_compatibility": "pending Qurrath controller confirmation",
    }
    write_json(METADATA_PATH, metadata)
    print(f"Training time (seconds): {training_seconds:.2f}")
    print(f"Epochs completed: {len(history.history['loss'])}")
    print(f"Validation accuracy: {validation_metrics['accuracy']:.6f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.6f}")
    print(f"Model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
