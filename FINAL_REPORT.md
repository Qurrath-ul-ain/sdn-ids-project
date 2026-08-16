# SDN Intrusion Detection System — Final Project Report

## Overview

This project implements a temporal CNN+LSTM+Attention classifier for four-class
network intrusion detection using the CSE-CIC-IDS2018 dataset.

---

## 1. Baseline Experiment

### 1.1 Methodology

**Data source:** Reduced CICFlowMeter CSV exports from four days of the
CSE-CIC-IDS2018 dataset.

**Temporal window type:** Chronological surrogate windows. Rows are
timestamp-ordered within each source file only. They are **NOT** reconstructed
network sessions. The source CSV files do not contain reliable session/flow
identifiers; row order is used as a temporal proxy.

**Label normalization:**

| Raw label              | Canonical label |
|------------------------|-----------------|
| Benign                 | Benign          |
| FTP-BruteForce         | Brute Force     |
| SSH-Bruteforce         | Brute Force     |
| Bot                    | Botnet          |
| Brute Force -Web       | Web Attack      |
| Brute Force -XSS       | Web Attack      |
| SQL Injection          | Web Attack      |

**Sampling:** 500 rows per canonical class (2,000 rows total), collected in
order of appearance from the source files.

**Features (in order):**

1. `destination_port`
2. `protocol`
3. `packet_count`  (tot_fwd_pkts + tot_bwd_pkts)
4. `byte_count`    (totlen_fwd_bytes + totlen_bwd_bytes)
5. `flow_duration_us`

**Sequence construction:**
- Sequence length: 10
- Stride: 1
- Shape: (10, 5)
- Label: label of the last row in the window
- Windows do not cross source-file boundaries

**Data split (stratified by class, applied before windowing):**
- Train: 70 % (350 rows / class)
- Validation: 15 % (75 rows / class)
- Test: 15 % (75 rows / class)

**Scaler:** StandardScaler fit on training rows only; applied to all splits.

### 1.2 Dataset Statistics

| Metric                        | Value     |
|-------------------------------|-----------|
| Total cleaned flow rows       | 2,000     |
| Classes (each)                | 500       |
| Train sequences               | 1,373     |
| Validation sequences          | 273       |
| Test sequences                | 273       |
| Sequence shape                | (10, 5)   |

### 1.3 Model Architecture

```
Input (10, 5)
→ Conv1D(32 filters, kernel=3, padding=same, activation=relu)
→ MaxPooling1D(pool_size=2)
→ LSTM(64 units, return_sequences=True)
→ Attention (custom; softmax-weighted sum over time steps)
→ Dense(32, relu)
→ Dropout(0.3)
→ Dense(4, softmax)
```

Total trainable parameters: **27,621**

Training:
- Optimizer: Adam (lr=0.001)
- Loss: sparse categorical cross-entropy
- Class weights: inverse-frequency balanced
- Early stopping: patience=3, restore_best_weights=True
- Max epochs: 20
- Batch size: 64

### 1.4 Evaluation Results (Test Set)

| Metric                | Value            |
|-----------------------|------------------|
| Accuracy              | 0.9744           |
| Macro Precision       | 0.9760           |
| Macro Recall          | 0.9767           |
| Macro F1              | 0.9752           |
| Weighted F1           | 0.9744           |
| Epochs completed      | 7                |
| Training time         | ~6.8 s           |

**Per-class metrics (test set):**

| Class       | Precision | Recall | F1     |
|-------------|-----------|--------|--------|
| Benign      | 0.9041    | 1.0000 | 0.9496 |
| Brute Force | 1.0000    | 1.0000 | 1.0000 |
| Botnet      | 1.0000    | 0.9067 | 0.9510 |
| Web Attack  | 1.0000    | 1.0000 | 1.0000 |

**Confusion Matrix (test set):**

```
                  Predicted
                  Benign  BruteForce  Botnet  WebAttack
Actual Benign      [ 66,       0,       0,       0 ]
Actual BruteForce  [  0,      66,       0,       0 ]
Actual Botnet      [  7,       0,      68,       0 ]
Actual WebAttack   [  0,       0,       0,      66 ]
```

(7 Botnet sequences mis-classified as Benign; all other classes perfect.)

### 1.5 Baseline Inference Result

Input: Sequence index 0 from the test split (shape (10,5), already normalised).

| Field           | Value                                      |
|-----------------|--------------------------------------------|
| True label      | Benign                                     |
| Prediction      | Benign                                     |
| Confidence      | 0.9994                                     |
| Correct         | Yes                                        |

Class probabilities:
- Benign:      0.9994
- Brute Force: 0.000059
- Botnet:      0.000278
- Web Attack:  0.000239

> **Important:** This result belongs entirely to the baseline surrogate-window
> experiment. It is **not** raw-PCAP inference.

---

## 2. Raw-PCAP Pipeline

### 2.1 Motivation

To replace the surrogate-window baseline with sequences constructed from
genuine bidirectional network flows, raw PCAP captures from the CSE-CIC-IDS2018
dataset were obtained and processed.

### 2.2 Source Files

| Day                    | UCAP file(s)            | Attack type  |
|------------------------|-------------------------|--------------|
| Wednesday-14-02-2018   | UCAP172.31.69.25        | Brute Force  |
| Friday-02-03-2018      | UCAP172.31.69.7/15/28   | Botnet       |
| Thursday-22-02-2018    | UCAP172.31.69.15/21/28  | Web Attack   |
| Friday-23-02-2018      | UCAP172.31.69.27/28     | Web Attack   |

Files were retrieved via selective HTTP range requests on the public S3 archive
(no full-archive download required).

### 2.3 Flow Extraction

A custom high-speed binary PCAP parser was implemented in Python using only
`struct` and `socket` (no Scapy dependency). For each packet:

- Parsed Ethernet → IPv4 → TCP/UDP header.
- Computed bidirectional flow key: `(min(src,dst), max(src,dst), protocol)`.
- Accumulated per-flow statistics: packet count, byte count, first/last timestamp.
- Applied UTC-correct attack-window filtering to assign labels.

### 2.4 Extracted Dataset

| Class       | Flows  |
|-------------|--------|
| Benign      | 23,588 |
| Brute Force | 14,117 |
| Web Attack  |    287 |
| Botnet      |     21 |
| **Total**   | **38,013** |

Flow partition (group-level stratified split):
- Train: 26,604 flows (26,581 groups)
- Validation: 5,699 flows (5,694 groups)
- Test: 5,710 flows (5,701 groups)

### 2.5 Mandated Temporal Grouping

The grouping key required for sequence construction is the **normalized
bidirectional 5-tuple**:

```
group_key = canonical_sort({Src IP, Src Port} ↔ {Dst IP, Dst Port}) + Protocol
```

Within each group, flows are sorted by Timestamp.

Sequence length = 10, stride = 1, input shape = (10, 5).

### 2.6 Group-Size Validation

Independent Python verification confirmed:

| Group size   | Count  |
|--------------|--------|
| Exactly 1    | 37,946 |
| 2 to 3       | 67     |
| 4 to 9       | 0      |
| ≥ 10         | **0**  |

**Maximum observed group size: 3**

Because client source ports are randomized per TCP/UDP connection, each
connection produces a unique 5-tuple. This is a fundamental property of
TCP/IP networking, not an implementation error.

### 2.7 Raw-PCAP Sequence Result

**Valid length-10 sequences produced: 0**

This is a verified data-coverage / grouping-constraint limitation.

### 2.8 Implication

The 96.7–97.4 % accuracy figures belong exclusively to the **baseline
surrogate-window experiment**.

> "The ~97 % accuracy was obtained from the baseline surrogate-window
> experiment. The raw-PCAP flow extraction and mandated temporal grouping
> pipeline was successfully validated; however, no communication group
> contained the required 10 flows, so a valid raw-PCAP sequence dataset
> could not be constructed. Consequently, raw-PCAP CNN+LSTM+Attention
> performance metrics were not produced."

---

## 3. Limitations

1. **Surrogate windows are not sessions.** The baseline sequences are
   chronological windows on CSV rows, not reconstructed TCP/UDP sessions.
   They satisfy the (10, 5) input requirement but should be clearly labeled
   as surrogate in any publication.

2. **Raw-PCAP grouping constraint.** The mandated 5-tuple grouping isolates
   each unique TCP/UDP connection. Most connections produce only 1 or 2
   packets visible in the host-level UCAP captures, resulting in groups of
   size 1. Acquiring a full-traffic PCAP (not a host-specific UCAP) would
   capture complete multi-packet flows and would likely yield groups of size ≥ 10.

3. **Class imbalance in raw flows.** Botnet (21 flows) and Web Attack
   (287 flows) are severely under-represented relative to Benign and Brute
   Force. Even with a larger PCAP source, the Botnet class would likely
   require careful sampling.

---

## 4. Artifact Locations

| Artifact                              | Path |
|---------------------------------------|------|
| Baseline sequences (train)            | `data/processed/temporal_sequences/train.npz` |
| Baseline sequences (validation)       | `data/processed/temporal_sequences/validation.npz` |
| Baseline sequences (test)             | `data/processed/temporal_sequences/test.npz` |
| Baseline scaler                       | `data/processed/temporal_sequences/scaler.json` |
| Baseline sequence metadata            | `data/metadata/temporal_sequence_metadata.json` |
| Baseline label mapping                | `data/metadata/temporal_label_mapping.json` |
| Trained model                         | `models/trained/cnn_lstm_attention.keras` |
| Evaluation results                    | `data/metadata/cnn_lstm_attention_results.json` |
| Baseline inference result             | `data/metadata/baseline_inference_result.json` |
| Raw-PCAP multi-day flows              | `data/raw/real_flows_multiday.csv` |
| Raw-PCAP sequence metadata            | `data/metadata/real_temporal_sequence_metadata.json` |
| Raw-PCAP real sequences (empty)       | `data/processed/real_temporal_sequences/` |
| Baseline inference script             | `run_baseline_inference.py` |
| Reproducibility commands              | `REPRODUCIBILITY.md` |
