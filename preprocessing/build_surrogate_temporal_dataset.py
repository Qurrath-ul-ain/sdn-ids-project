"""Build bounded surrogate windows from reduced CICFlowMeter CSVs.

These are chronological windows, not recovered network sessions: the source
files do not contain grouping identifiers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed" / "temporal_sequences"
METADATA_DIR = ROOT / "data" / "metadata"
FILES = [
    Path(r"C:\Users\Shinjini\Downloads\archive\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"),
    Path(r"C:\Users\Shinjini\Downloads\archive\Friday-23-02-2018_TrafficForML_CICFlowMeter.csv"),
    Path(r"C:\Users\Shinjini\Downloads\archive\Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv"),
    Path(r"C:\Users\Shinjini\Downloads\archive\Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"),
]
LABEL_MAP = {"Benign": 0, "Brute Force": 1, "Botnet": 2, "Web Attack": 3}
RAW_LABEL_MAP = {
    "Benign": "Benign",
    "Bot": "Botnet",
    "FTP-BruteForce": "Brute Force",
    "SSH-Bruteforce": "Brute Force",
    "Brute Force -Web": "Web Attack",
    "Brute Force -XSS": "Web Attack",
    "SQL Injection": "Web Attack",
}
LABEL_NAMES = list(LABEL_MAP)
FEATURE_NAMES = ["destination_port", "protocol", "packet_count", "byte_count", "flow_duration_us"]
SOURCE_COLUMNS = ["Dst Port", "Protocol", "Timestamp", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Label"]
MAX_PER_CLASS = 500
SEQUENCE_LENGTH = 10
STRIDE = 1
SEED = 42


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def collect_rows() -> pd.DataFrame:
    collected: dict[str, list[pd.DataFrame]] = {label: [] for label in LABEL_NAMES}
    totals = {label: 0 for label in LABEL_NAMES}
    for source in FILES:
        for chunk in pd.read_csv(source, usecols=SOURCE_COLUMNS, chunksize=100_000, low_memory=False):
            chunk["Label"] = chunk["Label"].astype("string").str.strip().map(RAW_LABEL_MAP)
            for label in LABEL_NAMES:
                remaining = MAX_PER_CLASS - totals[label]
                if remaining <= 0:
                    continue
                selected = chunk[chunk["Label"] == label].head(remaining).copy()
                if not selected.empty:
                    selected["source_file"] = source.name
                    collected[label].append(selected)
                    totals[label] += len(selected)
            if all(count >= MAX_PER_CLASS for count in totals.values()):
                break
        if all(count >= MAX_PER_CLASS for count in totals.values()):
            break
    missing = [label for label, count in totals.items() if count < MAX_PER_CLASS]
    if missing:
        raise RuntimeError(f"Could not collect the bounded quota for: {missing}")
    return pd.concat([frame for label in LABEL_NAMES for frame in collected[label]], ignore_index=True)


def main() -> None:
    rows = collect_rows()
    rows["Timestamp"] = pd.to_datetime(rows["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    numeric = pd.DataFrame({
        "destination_port": pd.to_numeric(rows["Dst Port"], errors="coerce"),
        "protocol": pd.to_numeric(rows["Protocol"], errors="coerce"),
        "packet_count": pd.to_numeric(rows["Tot Fwd Pkts"], errors="coerce") + pd.to_numeric(rows["Tot Bwd Pkts"], errors="coerce"),
        "byte_count": pd.to_numeric(rows["TotLen Fwd Pkts"], errors="coerce") + pd.to_numeric(rows["TotLen Bwd Pkts"], errors="coerce"),
        "flow_duration_us": pd.to_numeric(rows["Flow Duration"], errors="coerce"),
    })
    valid = numeric.notna().all(axis=1) & rows["Timestamp"].notna()
    rows = rows.loc[valid].reset_index(drop=True)
    numeric = numeric.loc[valid].reset_index(drop=True)
    numeric["Label"] = rows["Label"].map(LABEL_MAP).astype(int)
    numeric["source_file"] = rows["source_file"].to_numpy()
    numeric["Timestamp"] = rows["Timestamp"].to_numpy()

    rng = np.random.default_rng(SEED)
    split_codes = np.full(len(numeric), -1, dtype=np.int8)
    for label in range(len(LABEL_NAMES)):
        indexes = np.flatnonzero(numeric["Label"].to_numpy() == label)
        rng.shuffle(indexes)
        train_end = int(len(indexes) * 0.70)
        validation_end = train_end + int(len(indexes) * 0.15)
        split_codes[indexes[:train_end]] = 0
        split_codes[indexes[train_end:validation_end]] = 1
        split_codes[indexes[validation_end:]] = 2

    train_values = numeric.loc[split_codes == 0, FEATURE_NAMES].to_numpy(dtype=np.float64)
    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    scale[scale == 0] = 1.0
    numeric[FEATURE_NAMES] = (numeric[FEATURE_NAMES] - mean) / scale

    sequences: dict[str, tuple[list[np.ndarray], list[int]]] = {name: ([], []) for name in ("train", "validation", "test")}
    for split_code, split_name in enumerate(sequences):
        for source_file in numeric["source_file"].unique():
            indexes = np.flatnonzero((split_codes == split_code) & (numeric["source_file"].to_numpy() == source_file))
            ordered = numeric.iloc[indexes].sort_values("Timestamp")
            values = ordered[FEATURE_NAMES].to_numpy(dtype=np.float32)
            targets = ordered["Label"].to_numpy(dtype=np.int64)
            for start in range(0, len(values) - SEQUENCE_LENGTH + 1, STRIDE):
                sequences[split_name][0].append(values[start : start + SEQUENCE_LENGTH])
                sequences[split_name][1].append(int(targets[start + SEQUENCE_LENGTH - 1]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "source_files": [str(path) for path in FILES],
        "temporal_window_type": "chronological surrogate windows; not true sessions",
        "grouping": "none; source lacks reliable session identifiers",
        "sequence_length": SEQUENCE_LENGTH,
        "stride": STRIDE,
        "feature_order": FEATURE_NAMES,
        "input_shape": [SEQUENCE_LENGTH, len(FEATURE_NAMES)],
        "label_mapping": LABEL_MAP,
        "flow_rows_after_cleaning": int(len(numeric)),
        "flow_class_counts": {label: int((numeric["Label"] == index).sum()) for label, index in LABEL_MAP.items()},
        "sequence_counts": {name: len(values[1]) for name, values in sequences.items()},
        "sequence_class_counts": {name: {label: int(values[1].count(index)) for label, index in LABEL_MAP.items()} for name, values in sequences.items()},
        "split_method": "stratified flow-row split before window construction; windows do not cross partitions",
        "limitation": "Rows are timestamp-ordered within each source file only. They must not be interpreted as reconstructed network sessions.",
    }
    for split_name, (values, targets) in sequences.items():
        np.savez_compressed(OUT_DIR / f"{split_name}.npz", X=np.asarray(values, dtype=np.float32), y=np.asarray(targets, dtype=np.int64))
    write_json(OUT_DIR / "scaler.json", {"feature_order": FEATURE_NAMES, "mean": mean.tolist(), "scale": scale.tolist()})
    write_json(METADATA_DIR / "temporal_sequence_metadata.json", report)
    write_json(METADATA_DIR / "temporal_label_mapping.json", LABEL_MAP)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
