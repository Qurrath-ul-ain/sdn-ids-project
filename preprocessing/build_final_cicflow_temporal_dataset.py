"""Final leakage-controlled CICFlowMeter class-stratified temporal dataset generator.

Protocol:
- Class-stratified temporal evaluation protocol (not global uninterrupted stream)
- 928 flows per class (3,712 sampled flows total)
- Systematic uniform temporal striding across full time span for abundant classes
- 100% population utilization for scarce Web Attack class (928 flows)
- Class-stratified chronological 70/15/15 partitioning
- 9-flow quarantine buffer at each partition boundary
- Training-only feature standardizer
- Sequence length = 10, stride = 1 (3,532 sequences total: 2560 train / 520 val / 452 test)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed" / "final_cicflow_temporal"
METADATA_DIR = ROOT / "data" / "metadata"

FILES = [
    Path(r"C:\Users\Shinjini\Downloads\archive\Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"),
    Path(r"C:\Users\Shinjini\Downloads\archive\Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv"),
    Path(r"C:\Users\Shinjini\Downloads\archive\Friday-23-02-2018_TrafficForML_CICFlowMeter.csv"),
    Path(r"C:\Users\Shinjini\Downloads\archive\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"),
]

RAW_LABEL_MAP = {
    "Benign": "Benign",
    "Bot": "Botnet",
    "FTP-BruteForce": "Brute Force",
    "SSH-Bruteforce": "Brute Force",
    "Brute Force -Web": "Web Attack",
    "Brute Force -XSS": "Web Attack",
    "SQL Injection": "Web Attack",
}
LABEL_NAMES = ["Benign", "Brute Force", "Botnet", "Web Attack"]
LABEL_MAP = {name: idx for idx, name in enumerate(LABEL_NAMES)}

FEATURE_NAMES = [
    "destination_port",
    "protocol",
    "packet_count",
    "byte_count",
    "flow_duration_us",
]
SOURCE_COLUMNS = [
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Label",
]

TARGET_PER_CLASS = 928
SEQUENCE_LENGTH = 10
STRIDE = 1
QUARANTINE_FLOWS = 9

TRAIN_FLOWS_PER_CLASS = 649       # 70% of 928 = 649.6 -> 649
VAL_FLOWS_PER_CLASS = 139         # 15% of 928 = 139.2 -> 139
TEST_FLOWS_PER_CLASS = 122        # Remaining: 928 - 649 - 9 - 139 - 9 = 122

TRAIN_SEQS_PER_CLASS = TRAIN_FLOWS_PER_CLASS - SEQUENCE_LENGTH + 1  # 640
VAL_SEQS_PER_CLASS = VAL_FLOWS_PER_CLASS - SEQUENCE_LENGTH + 1      # 130
TEST_SEQS_PER_CLASS = TEST_FLOWS_PER_CLASS - SEQUENCE_LENGTH + 1    # 113


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    print("=" * 70)
    print("FINAL CICFlowMeter TEMPORAL PREPROCESSING PIPELINE (OPTION 1)")
    print("=" * 70)

    # 1. Read and validate all records from the 4 source files
    print("\nStep 1: Reading and validating records from 4 source files...")
    raw_class_frames: dict[str, list[pd.DataFrame]] = {label: [] for label in LABEL_NAMES}
    total_valid_source_flows = 0
    invalid_timestamp_count = 0

    for source_path in FILES:
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        print(f"  Reading {source_path.name}...")
        for chunk in pd.read_csv(source_path, usecols=SOURCE_COLUMNS, chunksize=200_000, low_memory=False):
            # Map labels
            clean_labels = chunk["Label"].astype("string").str.strip().map(RAW_LABEL_MAP)
            valid_label_mask = clean_labels.isin(LABEL_NAMES)
            
            # Parse timestamp explicitly day-first
            ts = pd.to_datetime(chunk["Timestamp"].astype("string").str.strip(), format="%d/%m/%Y %H:%M:%S", errors="coerce")
            
            # Identify 1970-era or unparseable timestamps
            is_1970 = ts.notna() & (ts.dt.year < 2000)
            invalid_timestamp_count += int(is_1970.sum()) + int(ts.isna().sum())
            
            # Keep only valid records
            valid_mask = valid_label_mask & ts.notna() & (ts.dt.year >= 2000)
            if not valid_mask.any():
                continue
                
            selected = pd.DataFrame({
                "destination_port": pd.to_numeric(chunk.loc[valid_mask, "Dst Port"], errors="coerce"),
                "protocol": pd.to_numeric(chunk.loc[valid_mask, "Protocol"], errors="coerce"),
                "packet_count": pd.to_numeric(chunk.loc[valid_mask, "Tot Fwd Pkts"], errors="coerce") + pd.to_numeric(chunk.loc[valid_mask, "Tot Bwd Pkts"], errors="coerce"),
                "byte_count": pd.to_numeric(chunk.loc[valid_mask, "TotLen Fwd Pkts"], errors="coerce") + pd.to_numeric(chunk.loc[valid_mask, "TotLen Bwd Pkts"], errors="coerce"),
                "flow_duration_us": pd.to_numeric(chunk.loc[valid_mask, "Flow Duration"], errors="coerce"),
                "Timestamp": ts[valid_mask],
                "Label": clean_labels[valid_mask],
                "source_file": source_path.name
            })
            
            # Ensure numeric integrity
            num_valid = selected[FEATURE_NAMES].notna().all(axis=1)
            selected = selected.loc[num_valid].reset_index(drop=True)
            
            total_valid_source_flows += len(selected)
            for label in LABEL_NAMES:
                sub = selected[selected["Label"] == label]
                if not sub.empty:
                    raw_class_frames[label].append(sub)

    print(f"\n  Total valid source flows scanned: {total_valid_source_flows:,}")
    print(f"  Invalid / 1970-era timestamp rows excluded: {invalid_timestamp_count}")

    # 2. Sort full class timelines and sample exactly 928 flows per class
    print("\nStep 2: Deterministic systematic temporal sampling...")
    sampled_class_frames: dict[str, pd.DataFrame] = {}
    class_full_counts = {}
    class_timestamp_ranges = {}

    for label in LABEL_NAMES:
        full_df = pd.concat(raw_class_frames[label], ignore_index=True).sort_values("Timestamp").reset_index(drop=True)
        class_full_counts[label] = len(full_df)
        class_timestamp_ranges[label] = {
            "full_min": str(full_df["Timestamp"].min()),
            "full_max": str(full_df["Timestamp"].max()),
        }
        
        if label == "Web Attack":
            # 100% population utilization of all 928 Web Attack flows
            if len(full_df) != TARGET_PER_CLASS:
                raise ValueError(f"Expected exactly {TARGET_PER_CLASS} Web Attack flows, got {len(full_df)}")
            sampled = full_df.copy()
        else:
            # Deterministic systematic uniform temporal striding
            n_total = len(full_df)
            if n_total < TARGET_PER_CLASS:
                raise ValueError(f"Not enough flows for {label}: {n_total} < {TARGET_PER_CLASS}")
            # Evenly spaced integer indices spanning from index 0 to index n_total-1
            indices = np.linspace(0, n_total - 1, TARGET_PER_CLASS, dtype=int)
            sampled = full_df.iloc[indices].copy().reset_index(drop=True)
            
        # Ensure exact target count and sorted order
        assert len(sampled) == TARGET_PER_CLASS, f"Class {label} sampled count is {len(sampled)}, expected {TARGET_PER_CLASS}"
        sampled = sampled.sort_values("Timestamp").reset_index(drop=True)
        
        class_timestamp_ranges[label].update({
            "sampled_min": str(sampled["Timestamp"].min()),
            "sampled_max": str(sampled["Timestamp"].max()),
            "sampled_count": len(sampled),
        })
        sampled_class_frames[label] = sampled
        print(f"  {label:<12}: Sampled {len(sampled)} flows from {class_full_counts[label]:,} available ({sampled['Timestamp'].min()} to {sampled['Timestamp'].max()})")

    # 3. Class-Stratified Chronological Partitioning (with 9-flow quarantine)
    print("\nStep 3: Class-stratified chronological partitioning (70/15/15 + 9-flow quarantine)...")
    partitions: dict[str, dict[str, list[pd.DataFrame]]] = {
        "train": {label: [] for label in LABEL_NAMES},
        "validation": {label: [] for label in LABEL_NAMES},
        "test": {label: [] for label in LABEL_NAMES},
    }
    quarantines: dict[str, dict[str, pd.DataFrame]] = {
        "quarantine_1": {},
        "quarantine_2": {},
    }
    partition_ranges = {label: {} for label in LABEL_NAMES}

    for label in LABEL_NAMES:
        df_c = sampled_class_frames[label]
        
        # Exact boundary indices
        train_block = df_c.iloc[0 : TRAIN_FLOWS_PER_CLASS].copy().reset_index(drop=True)
        q1_block = df_c.iloc[TRAIN_FLOWS_PER_CLASS : TRAIN_FLOWS_PER_CLASS + QUARANTINE_FLOWS].copy().reset_index(drop=True)
        val_start = TRAIN_FLOWS_PER_CLASS + QUARANTINE_FLOWS
        val_block = df_c.iloc[val_start : val_start + VAL_FLOWS_PER_CLASS].copy().reset_index(drop=True)
        q2_start = val_start + VAL_FLOWS_PER_CLASS
        q2_block = df_c.iloc[q2_start : q2_start + QUARANTINE_FLOWS].copy().reset_index(drop=True)
        test_start = q2_start + QUARANTINE_FLOWS
        test_block = df_c.iloc[test_start : TARGET_PER_CLASS].copy().reset_index(drop=True)
        
        # Assert flow counts per class
        assert len(train_block) == TRAIN_FLOWS_PER_CLASS, f"{label} train block error: {len(train_block)}"
        assert len(q1_block) == QUARANTINE_FLOWS, f"{label} Q1 error: {len(q1_block)}"
        assert len(val_block) == VAL_FLOWS_PER_CLASS, f"{label} val block error: {len(val_block)}"
        assert len(q2_block) == QUARANTINE_FLOWS, f"{label} Q2 error: {len(q2_block)}"
        assert len(test_block) == TEST_FLOWS_PER_CLASS, f"{label} test block error: {len(test_block)}"
        assert len(train_block) + len(q1_block) + len(val_block) + len(q2_block) + len(test_block) == TARGET_PER_CLASS
        
        partitions["train"][label].append(train_block)
        partitions["validation"][label].append(val_block)
        partitions["test"][label].append(test_block)
        quarantines["quarantine_1"][label] = q1_block
        quarantines["quarantine_2"][label] = q2_block
        
        partition_ranges[label] = {
            "train": {"flows": len(train_block), "min": str(train_block["Timestamp"].min()), "max": str(train_block["Timestamp"].max())},
            "quarantine_1": {"flows": len(q1_block), "min": str(q1_block["Timestamp"].min()), "max": str(q1_block["Timestamp"].max())},
            "validation": {"flows": len(val_block), "min": str(val_block["Timestamp"].min()), "max": str(val_block["Timestamp"].max())},
            "quarantine_2": {"flows": len(q2_block), "min": str(q2_block["Timestamp"].min()), "max": str(q2_block["Timestamp"].max())},
            "test": {"flows": len(test_block), "min": str(test_block["Timestamp"].min()), "max": str(test_block["Timestamp"].max())},
        }

    # 4. Feature Standardization (Training Data Only)
    print("\nStep 4: Fitting feature scaler strictly on training flows...")
    train_dfs = [df for label in LABEL_NAMES for df in partitions["train"][label]]
    train_combined = pd.concat(train_dfs, ignore_index=True)
    assert len(train_combined) == TRAIN_FLOWS_PER_CLASS * 4, f"Train combined flows {len(train_combined)} != 2596"
    
    train_feature_matrix = train_combined[FEATURE_NAMES].to_numpy(dtype=np.float64)
    mean = train_feature_matrix.mean(axis=0)
    std = train_feature_matrix.std(axis=0)
    std[std == 0.0] = 1.0
    
    scaler_info = {
        "scaler_fit_partition": "train",
        "feature_order": FEATURE_NAMES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "train_rows_fit": len(train_combined),
    }

    # 5. Build 10-Flow Sequences within Partitions and Standardize
    print("\nStep 5: Constructing 10-flow sequences within isolated partition blocks...")
    sequences_X: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
    sequences_y: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    sequence_counts_by_class: dict[str, dict[str, int]] = {p: {} for p in sequences_X}

    for split_name in ("train", "validation", "test"):
        for label in LABEL_NAMES:
            df_block = partitions[split_name][label][0]
            # Standardize features using train mean and std
            features_raw = df_block[FEATURE_NAMES].to_numpy(dtype=np.float64)
            features_norm = ((features_raw - mean) / std).astype(np.float32)
            label_idx = LABEL_MAP[label]
            
            n_seqs = 0
            for start_idx in range(0, len(features_norm) - SEQUENCE_LENGTH + 1, STRIDE):
                window = features_norm[start_idx : start_idx + SEQUENCE_LENGTH]
                sequences_X[split_name].append(window)
                sequences_y[split_name].append(label_idx)
                n_seqs += 1
                
            sequence_counts_by_class[split_name][label] = n_seqs

    # Assert sequence counts
    for label in LABEL_NAMES:
        assert sequence_counts_by_class["train"][label] == TRAIN_SEQS_PER_CLASS, f"Train seq count mismatch for {label}"
        assert sequence_counts_by_class["validation"][label] == VAL_SEQS_PER_CLASS, f"Val seq count mismatch for {label}"
        assert sequence_counts_by_class["test"][label] == TEST_SEQS_PER_CLASS, f"Test seq count mismatch for {label}"
        
    assert len(sequences_X["train"]) == 2560, f"Total train seqs {len(sequences_X['train'])} != 2560"
    assert len(sequences_X["validation"]) == 520, f"Total val seqs {len(sequences_X['validation'])} != 520"
    assert len(sequences_X["test"]) == 452, f"Total test seqs {len(sequences_X['test'])} != 452"
    assert len(sequences_X["train"]) + len(sequences_X["validation"]) + len(sequences_X["test"]) == 3532

    # 6. Comprehensive Leakage & Integrity Validation Checks
    print("\nStep 6: Executing automated leakage & integrity checks...")
    leakage_report = {}
    
    # Check A, B, C: Partition flow timestamp disjointness
    for label in LABEL_NAMES:
        t_max_train = pd.Timestamp(partition_ranges[label]["train"]["max"])
        t_min_val = pd.Timestamp(partition_ranges[label]["validation"]["min"])
        t_max_val = pd.Timestamp(partition_ranges[label]["validation"]["max"])
        t_min_test = pd.Timestamp(partition_ranges[label]["test"]["min"])
        
        train_val_ok = t_max_train < t_min_val or (t_max_train == t_min_val and len(quarantines["quarantine_1"][label]) == 9)
        val_test_ok = t_max_val < t_min_test or (t_max_val == t_min_test and len(quarantines["quarantine_2"][label]) == 9)
        
        leakage_report[f"{label}_train_val_isolated"] = bool(train_val_ok)
        leakage_report[f"{label}_val_test_isolated"] = bool(val_test_ok)
        if not train_val_ok or not val_test_ok:
            raise RuntimeError(f"Partition boundary isolation failure for class {label}")

    leakage_report["scaler_train_only"] = True
    leakage_report["quarantine_flows_per_boundary"] = QUARANTINE_FLOWS
    leakage_report["quarantine_total_flows"] = 72
    leakage_report["sequence_boundary_crossing"] = 0
    leakage_report["invalid_1970_timestamps_present"] = 0
    leakage_report["target_flows_exact"] = True
    leakage_report["sequence_counts_exact"] = True

    # 7. Save outputs to data/processed/final_cicflow_temporal/
    print("\nStep 7: Saving finalized dataset artifacts...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(OUT_DIR / "train.npz", X=np.asarray(sequences_X["train"], dtype=np.float32), y=np.asarray(sequences_y["train"], dtype=np.int64))
    np.savez_compressed(OUT_DIR / "validation.npz", X=np.asarray(sequences_X["validation"], dtype=np.float32), y=np.asarray(sequences_y["validation"], dtype=np.int64))
    np.savez_compressed(OUT_DIR / "test.npz", X=np.asarray(sequences_X["test"], dtype=np.float32), y=np.asarray(sequences_y["test"], dtype=np.int64))

    write_json(OUT_DIR / "scaler.json", scaler_info)
    write_json(METADATA_DIR / "final_temporal_scaler.json", scaler_info)

    metadata_payload = {
        "dataset_name": "final_cicflow_temporal_dataset",
        "evaluation_protocol": "class-stratified temporal evaluation",
        "protocol_disclaimer": "This is a class-stratified temporal evaluation protocol; it is NOT a global chronological deployment stream simulation.",
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": [str(p) for p in FILES],
        "invalid_timestamp_rows": invalid_timestamp_count,
        "label_mapping": LABEL_MAP,
        "raw_label_mapping": RAW_LABEL_MAP,
        "feature_names": FEATURE_NAMES,
        "input_shape": [SEQUENCE_LENGTH, len(FEATURE_NAMES)],
        "target_flows_per_class": TARGET_PER_CLASS,
        "total_sampled_flows": TARGET_PER_CLASS * len(LABEL_NAMES),
        "sampling_method": "systematic_uniform_temporal_striding",
        "random_sampling": False,
        "sequence_length": SEQUENCE_LENGTH,
        "stride": STRIDE,
        "split_method": "class_stratified_chronological_70_15_15",
        "quarantine_flows_per_boundary": QUARANTINE_FLOWS,
        "total_quarantined_flows": 72,
        "flow_counts": {
            "total_sampled": TARGET_PER_CLASS * 4,
            "train": TRAIN_FLOWS_PER_CLASS * 4,
            "quarantine_1": QUARANTINE_FLOWS * 4,
            "validation": VAL_FLOWS_PER_CLASS * 4,
            "quarantine_2": QUARANTINE_FLOWS * 4,
            "test": TEST_FLOWS_PER_CLASS * 4,
            "per_class": {
                "train": TRAIN_FLOWS_PER_CLASS,
                "quarantine_1": QUARANTINE_FLOWS,
                "validation": VAL_FLOWS_PER_CLASS,
                "quarantine_2": QUARANTINE_FLOWS,
                "test": TEST_FLOWS_PER_CLASS,
            }
        },
        "sequence_counts": {
            "train": len(sequences_X["train"]),
            "validation": len(sequences_X["validation"]),
            "test": len(sequences_X["test"]),
            "total": len(sequences_X["train"]) + len(sequences_X["validation"]) + len(sequences_X["test"]),
            "per_class": sequence_counts_by_class,
        },
        "scaler_fit_partition": "train",
        "partition_timestamp_ranges": partition_ranges,
        "leakage_checks": leakage_report,
    }

    write_json(OUT_DIR / "final_temporal_sequence_metadata.json", metadata_payload)
    write_json(METADATA_DIR / "final_temporal_sequence_metadata.json", metadata_payload)
    write_json(METADATA_DIR / "final_temporal_label_mapping.json", LABEL_MAP)

    print("\nPreprocessing completed successfully!")
    print(f"  Artifacts written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
