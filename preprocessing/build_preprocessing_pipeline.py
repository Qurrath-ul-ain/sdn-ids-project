"""Build leakage-free, controller-pending preprocessing splits for SDN-IDS."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "four_class_dataset.csv"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
PREPROCESSING_DIR = PROJECT_ROOT / "models" / "preprocessing"

RANDOM_SEED = 42
CHUNK_SIZE = 100_000
LABEL_COLUMN = "Label"
SOURCE_COLUMNS = [
    "Dst Port",
    "Protocol",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Flow Duration",
    LABEL_COLUMN,
]
FEATURE_NAMES = [
    "destination_port",
    "protocol",
    "packet_count",
    "byte_count",
    "flow_duration_us",
]
LABEL_MAPPING = {
    "Benign": 0,
    "Brute Force": 1,
    "Botnet": 2,
    "Web Attack": 3,
}
SPLIT_NAMES = ("train", "validation", "test")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2)
        output_file.write("\n")


def canonicalize_chunk(chunk: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    labels = chunk[LABEL_COLUMN].astype("string").str.strip()
    features = pd.DataFrame(
        {
            "destination_port": pd.to_numeric(chunk["Dst Port"], errors="coerce"),
            "protocol": pd.to_numeric(chunk["Protocol"], errors="coerce"),
            "packet_count": pd.to_numeric(chunk["Tot Fwd Pkts"], errors="coerce")
            + pd.to_numeric(chunk["Tot Bwd Pkts"], errors="coerce"),
            "byte_count": pd.to_numeric(chunk["TotLen Fwd Pkts"], errors="coerce")
            + pd.to_numeric(chunk["TotLen Bwd Pkts"], errors="coerce"),
            "flow_duration_us": pd.to_numeric(chunk["Flow Duration"], errors="coerce"),
        }
    )
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, labels


def valid_row_mask(features: pd.DataFrame, labels: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_labels = labels.isin(LABEL_MAPPING).to_numpy(dtype=bool)
    valid_features = features.notna().all(axis=1).to_numpy(dtype=bool)
    return valid_labels & valid_features, valid_labels, valid_features


def split_sizes(row_count: int) -> tuple[int, int, int]:
    train_count = int(np.floor(row_count * 0.70))
    validation_count = int(np.floor(row_count * 0.15))
    return train_count, validation_count, row_count - train_count - validation_count


def build_assignments(class_counts: Counter[str]) -> dict[str, np.ndarray]:
    assignments: dict[str, np.ndarray] = {}
    for offset, label in enumerate(LABEL_MAPPING):
        row_count = class_counts[label]
        train_count, validation_count, _ = split_sizes(row_count)
        split_codes = np.full(row_count, 2, dtype=np.uint8)
        split_codes[:train_count] = 0
        split_codes[train_count : train_count + validation_count] = 1
        rng = np.random.default_rng(RANDOM_SEED + offset)
        rng.shuffle(split_codes)
        assignments[label] = split_codes
    return assignments


def assign_splits(
    labels: pd.Series,
    valid_mask: np.ndarray,
    assignments: dict[str, np.ndarray],
    positions: dict[str, int],
) -> np.ndarray:
    codes = np.full(len(labels), -1, dtype=np.int8)
    label_values = labels.to_numpy()
    for label in LABEL_MAPPING:
        row_indexes = np.flatnonzero(valid_mask & (label_values == label))
        start = positions[label]
        end = start + len(row_indexes)
        codes[row_indexes] = assignments[label][start:end]
        positions[label] = end
    return codes


def inspect_valid_rows() -> tuple[Counter[str], dict[str, int]]:
    class_counts: Counter[str] = Counter()
    report = {
        "rows_read": 0,
        "invalid_label_rows": 0,
        "invalid_feature_rows": 0,
        "rows_dropped": 0,
    }
    for chunk in pd.read_csv(DATASET_PATH, usecols=SOURCE_COLUMNS, chunksize=CHUNK_SIZE, low_memory=False):
        features, labels = canonicalize_chunk(chunk)
        valid_mask, valid_labels, valid_features = valid_row_mask(features, labels)
        report["rows_read"] += len(chunk)
        report["invalid_label_rows"] += int((~valid_labels).sum())
        report["invalid_feature_rows"] += int((~valid_features).sum())
        report["rows_dropped"] += int((~valid_mask).sum())
        class_counts.update(labels[valid_mask].tolist())
    return class_counts, report


def fit_scaler(assignments: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    positions = {label: 0 for label in LABEL_MAPPING}
    feature_sum = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    feature_sum_squares = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    train_rows = 0

    for chunk in pd.read_csv(DATASET_PATH, usecols=SOURCE_COLUMNS, chunksize=CHUNK_SIZE, low_memory=False):
        features, labels = canonicalize_chunk(chunk)
        valid_mask, _, _ = valid_row_mask(features, labels)
        codes = assign_splits(labels, valid_mask, assignments, positions)
        train_values = features.to_numpy(dtype=np.float64, copy=False)[codes == 0]
        feature_sum += train_values.sum(axis=0)
        feature_sum_squares += np.square(train_values).sum(axis=0)
        train_rows += len(train_values)

    mean = feature_sum / train_rows
    variance = np.maximum(feature_sum_squares / train_rows - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale == 0] = 1.0
    return mean, scale, train_rows


def write_scaled_splits(assignments: dict[str, np.ndarray], mean: np.ndarray, scale: np.ndarray) -> dict[str, Counter[str]]:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    positions = {label: 0 for label in LABEL_MAPPING}
    split_distributions = {name: Counter() for name in SPLIT_NAMES}
    has_written = {name: False for name in SPLIT_NAMES}

    for chunk in pd.read_csv(DATASET_PATH, usecols=SOURCE_COLUMNS, chunksize=CHUNK_SIZE, low_memory=False):
        features, labels = canonicalize_chunk(chunk)
        valid_mask, _, _ = valid_row_mask(features, labels)
        codes = assign_splits(labels, valid_mask, assignments, positions)
        values = features.to_numpy(dtype=np.float64, copy=False)

        for split_code, split_name in enumerate(SPLIT_NAMES):
            split_mask = codes == split_code
            if not split_mask.any():
                continue
            transformed = (values[split_mask] - mean) / scale
            output = pd.DataFrame(transformed, columns=FEATURE_NAMES)
            output[LABEL_COLUMN] = labels.to_numpy()[split_mask]
            output.to_csv(
                SPLITS_DIR / f"{split_name}.csv",
                mode="a" if has_written[split_name] else "w",
                header=not has_written[split_name],
                index=False,
            )
            has_written[split_name] = True
            split_distributions[split_name].update(output[LABEL_COLUMN].tolist())
    return split_distributions


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATASET_PATH}")

    available_columns = set(pd.read_csv(DATASET_PATH, nrows=0).columns)
    missing_columns = [column for column in SOURCE_COLUMNS if column not in available_columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    class_counts, validation_report = inspect_valid_rows()
    missing_labels = [label for label in LABEL_MAPPING if class_counts[label] == 0]
    if missing_labels:
        raise ValueError(f"No valid rows found for required labels: {missing_labels}")

    assignments = build_assignments(class_counts)
    mean, scale, train_rows = fit_scaler(assignments)
    split_distributions = write_scaled_splits(assignments, mean, scale)

    for split_name, distribution in split_distributions.items():
        absent_labels = [label for label in LABEL_MAPPING if distribution[label] == 0]
        if absent_labels:
            raise RuntimeError(f"{split_name} split is missing labels: {absent_labels}")

    split_counts = {
        split_name: int(sum(distribution.values()))
        for split_name, distribution in split_distributions.items()
    }
    write_json(METADATA_DIR / "feature_list.json", FEATURE_NAMES)
    write_json(METADATA_DIR / "label_mapping.json", LABEL_MAPPING)
    write_json(
        PREPROCESSING_DIR / "scaler.json",
        {
            "method": "standard scaling",
            "fit_split": "train",
            "feature_order": FEATURE_NAMES,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
        },
    )
    write_json(
        PREPROCESSING_DIR / "cleaning_policy.json",
        {
            "numeric_conversion": "pandas.to_numeric(errors='coerce')",
            "invalid_value_policy": "drop rows with missing, non-numeric, or infinite selected features",
            "label_policy": "drop rows outside the required four labels",
            "controller_compatibility": "pending Qurrath controller confirmation",
        },
    )
    write_json(
        METADATA_DIR / "preprocessing_report.json",
        {
            **validation_report,
            "valid_rows": int(sum(class_counts.values())),
            "random_seed": RANDOM_SEED,
            "feature_order": FEATURE_NAMES,
            "label_mapping": LABEL_MAPPING,
            "class_distribution_before_split": dict(class_counts),
            "split_row_counts": split_counts,
            "split_class_distributions": {
                split_name: dict(distribution)
                for split_name, distribution in split_distributions.items()
            },
            "all_classes_present_in_each_split": True,
            "scaler_fit_rows": train_rows,
            "scaler_fit_split": "train",
            "controller_compatibility": "pending Qurrath controller confirmation",
        },
    )

    print("PREPROCESSING COMPLETE")
    print(f"Valid rows: {sum(class_counts.values()):,}")
    print(f"Invalid/dropped rows: {validation_report['rows_dropped']:,}")
    for split_name in SPLIT_NAMES:
        print(f"{split_name.title()} rows: {split_counts[split_name]:,}")
        for label in LABEL_MAPPING:
            print(f"  {label}: {split_distributions[split_name][label]:,}")


if __name__ == "__main__":
    main()
