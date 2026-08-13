"""Train and evaluate the four-class SDN-IDS Random Forest baseline."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
MODEL_DIR = PROJECT_ROOT / "models" / "trained"
FEATURE_LIST_PATH = METADATA_DIR / "feature_list.json"
LABEL_MAPPING_PATH = METADATA_DIR / "label_mapping.json"
MODEL_PATH = MODEL_DIR / "random_forest.joblib"
METADATA_PATH = METADATA_DIR / "random_forest_metadata.json"
RANDOM_SEED = 42


def read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2)
        output_file.write("\n")


def load_split(split_name: str, feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    split_path = SPLITS_DIR / f"{split_name}.csv"
    required_columns = [*feature_names, "Label"]
    frame = pd.read_csv(split_path, usecols=required_columns, low_memory=False)
    if frame.columns.tolist() != required_columns:
        raise ValueError(
            f"{split_path} columns do not match the canonical feature order: "
            f"expected {required_columns}, got {frame.columns.tolist()}"
        )
    return frame[feature_names], frame["Label"]


def metrics(y_true: pd.Series, y_pred: object, labels: list[str]) -> dict[str, object]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def main() -> None:
    feature_names = read_json(FEATURE_LIST_PATH)
    label_mapping = read_json(LABEL_MAPPING_PATH)
    if not isinstance(feature_names, list) or not all(isinstance(item, str) for item in feature_names):
        raise ValueError("feature_list.json must contain an ordered list of feature names")
    if not isinstance(label_mapping, dict):
        raise ValueError("label_mapping.json must contain a label mapping")
    labels = list(label_mapping)

    train_features, train_labels = load_split("train", feature_names)
    validation_features, validation_labels = load_split("validation", feature_names)

    configuration = {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**configuration)

    print(f"Training Random Forest on {len(train_features):,} rows...")
    training_started = time.perf_counter()
    model.fit(train_features, train_labels)
    training_seconds = time.perf_counter() - training_started

    validation_predictions = model.predict(validation_features)
    validation_metrics = metrics(validation_labels, validation_predictions, labels)

    # The configuration is fixed before loading the held-out test split.
    test_features, test_labels = load_split("test", feature_names)
    test_predictions = model.predict(test_features)
    test_metrics = metrics(test_labels, test_predictions, labels)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    model_metadata = {
        "model_type": "RandomForestClassifier",
        "configuration": configuration,
        "feature_order": feature_names,
        "label_mapping": label_mapping,
        "training_rows": len(train_features),
        "validation_rows": len(validation_features),
        "test_rows": len(test_features),
        "training_seconds": training_seconds,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "feature_importance": {
            feature: float(importance)
            for feature, importance in zip(feature_names, model.feature_importances_, strict=True)
        },
        "selection_policy": "Fixed baseline configuration; validation was evaluated before one held-out test evaluation.",
        "controller_compatibility": "pending Qurrath controller confirmation",
    }
    write_json(METADATA_PATH, model_metadata)

    print(f"Training time (seconds): {training_seconds:.2f}")
    print(f"Validation accuracy: {validation_metrics['accuracy']:.6f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.6f}")
    print(f"Model: {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
