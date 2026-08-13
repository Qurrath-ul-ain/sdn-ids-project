"""Tune and evaluate probability-level RF + DNN fusion."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
RF_PATH = PROJECT_ROOT / "models" / "trained" / "random_forest.joblib"
DNN_PATH = PROJECT_ROOT / "models" / "trained" / "dnn.keras"
OUTPUT_PATH = METADATA_DIR / "hybrid_metadata.json"
FEATURE_LIST_PATH = METADATA_DIR / "feature_list.json"
LABEL_MAPPING_PATH = METADATA_DIR / "label_mapping.json"
WEIGHTS = [(weight / 10, 1 - weight / 10) for weight in range(1, 10)]


def read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2)
        output_file.write("\n")


def load_split(split_name: str, feature_names: list[str], label_mapping: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    columns = [*feature_names, "Label"]
    frame = pd.read_csv(SPLITS_DIR / f"{split_name}.csv", usecols=columns, low_memory=False)
    if frame.columns.tolist() != columns:
        raise ValueError(f"{split_name} split does not preserve canonical feature order")
    labels = frame.pop("Label").map(label_mapping)
    if labels.isna().any():
        raise ValueError(f"{split_name} split contains an unknown label")
    return frame.to_numpy(dtype=np.float32), labels.to_numpy(dtype=np.int32)


def reorder_probabilities(probabilities: np.ndarray, source_order: list[str], target_order: list[str]) -> np.ndarray:
    if set(source_order) != set(target_order) or len(source_order) != len(target_order):
        raise ValueError(f"Probability class mismatch: source={source_order}, target={target_order}")
    indexes = [source_order.index(label) for label in target_order]
    return probabilities[:, indexes]


def score(y_true: np.ndarray, probabilities: np.ndarray, labels: list[str]) -> dict[str, object]:
    predictions = probabilities.argmax(axis=1)
    report = classification_report(
        y_true,
        predictions,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "web_attack_precision": report["Web Attack"]["precision"],
        "web_attack_recall": report["Web Attack"]["recall"],
        "web_attack_f1": report["Web Attack"]["f1-score"],
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
            }
            for label in labels
        },
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=list(range(len(labels)))).tolist(),
    }


def main() -> None:
    feature_names = read_json(FEATURE_LIST_PATH)
    label_mapping = read_json(LABEL_MAPPING_PATH)
    labels = list(label_mapping)
    rf_metadata = read_json(METADATA_DIR / "random_forest_metadata.json")
    dnn_metadata = read_json(METADATA_DIR / "dnn_metadata.json")
    rf = joblib.load(RF_PATH)
    dnn = tf.keras.models.load_model(DNN_PATH)

    canonical_order = labels
    rf_order = list(rf.classes_)
    metadata_rf_order = list(rf_metadata["label_mapping"])
    metadata_dnn_order = list(dnn_metadata["label_mapping"])
    if set(rf_order) != set(metadata_rf_order):
        raise ValueError(f"RF model/metadata class set mismatch: {rf_order} vs {metadata_rf_order}")
    if metadata_dnn_order != canonical_order:
        raise ValueError(f"DNN metadata does not match canonical order: {metadata_dnn_order}")
    if list(rf_metadata["feature_order"]) != list(feature_names) or list(dnn_metadata["feature_order"]) != list(feature_names):
        raise ValueError("Base model feature order does not match feature_list.json")

    validation_features, validation_labels = load_split("validation", feature_names, label_mapping)
    rf_validation = reorder_probabilities(rf.predict_proba(validation_features), rf_order, canonical_order)
    dnn_validation = dnn.predict(validation_features, batch_size=8192, verbose=0)
    validation_results = []
    for rf_weight, dnn_weight in WEIGHTS:
        probabilities = rf_weight * rf_validation + dnn_weight * dnn_validation
        result = score(validation_labels, probabilities, labels)
        validation_results.append({"rf_weight": rf_weight, "dnn_weight": dnn_weight, **result})

    # Primary criterion is macro F1; ties use Web Attack F1, then precision.
    selected = max(
        validation_results,
        key=lambda result: (
            result["macro_f1"],
            result["web_attack_f1"],
            result["web_attack_precision"],
        ),
    )
    selected_rf_weight = selected["rf_weight"]
    selected_dnn_weight = selected["dnn_weight"]

    # Test is loaded and evaluated only after validation weight selection.
    test_features, test_labels = load_split("test", feature_names, label_mapping)
    rf_test = reorder_probabilities(rf.predict_proba(test_features), rf_order, canonical_order)
    dnn_test = dnn.predict(test_features, batch_size=8192, verbose=0)
    test_probabilities = selected_rf_weight * rf_test + selected_dnn_weight * dnn_test
    test_result = score(test_labels, test_probabilities, labels)

    metadata = {
        "model_type": "probability-level RF + DNN fusion",
        "feature_order": feature_names,
        "class_order": canonical_order,
        "base_models": {
            "random_forest": str(RF_PATH),
            "dnn": str(DNN_PATH),
        },
        "probability_formula": "rf_weight * rf_probabilities + dnn_weight * dnn_probabilities",
        "rf_source_class_order": rf_order,
        "dnn_source_class_order": metadata_dnn_order,
        "selected_weights": {
            "rf_weight": selected_rf_weight,
            "dnn_weight": selected_dnn_weight,
        },
        "selection_criterion": "Validation macro F1 primary; Web Attack F1 and precision as tie-breakers",
        "validation_results": validation_results,
        "test_result": test_result,
        "comparison": {
            "random_forest_test": rf_metadata["test_metrics"],
            "dnn_test": dnn_metadata["test_metrics"],
        },
        "controller_compatibility": "pending Qurrath controller confirmation",
    }
    write_json(OUTPUT_PATH, metadata)
    print(f"Selected weights: RF={selected_rf_weight:.1f}, DNN={selected_dnn_weight:.1f}")
    print(f"Validation macro F1: {selected['macro_f1']:.6f}")
    print(f"Test accuracy: {test_result['accuracy']:.6f}")
    print(f"Metadata: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
