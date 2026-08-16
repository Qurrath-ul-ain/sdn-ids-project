# Reproducibility Guide — SDN-IDS Project

All commands assume the working directory is the project root:
`C:\Users\Shinjini\sdn-ids-project`

The Python interpreter is:
`C:\ProgramData\Anaconda3\python.exe`

> **Note:** TF_USE_LEGACY_KERAS=1 must be set before running any script
> that imports TensorFlow. Use the `.bat` wrappers provided, or set the
> environment variable in your shell session first.

---

## 1. Baseline — Preprocessing

Reads raw CICFlowMeter CSVs, collects 500 rows per class, builds
(10,5) surrogate temporal sequences, fits a StandardScaler on training
rows only, and saves NPZ files plus metadata.

**Required input files (not included in repo):**
- `C:\Users\Shinjini\Downloads\archive\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv`
- `C:\Users\Shinjini\Downloads\archive\Friday-23-02-2018_TrafficForML_CICFlowMeter.csv`
- `C:\Users\Shinjini\Downloads\archive\Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv`
- `C:\Users\Shinjini\Downloads\archive\Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv`

```bat
@echo off
set TF_USE_LEGACY_KERAS=1
"C:\ProgramData\Anaconda3\python.exe" preprocessing\build_surrogate_temporal_dataset.py
```

**Outputs:**
- `data/processed/temporal_sequences/train.npz`
- `data/processed/temporal_sequences/validation.npz`
- `data/processed/temporal_sequences/test.npz`
- `data/processed/temporal_sequences/scaler.json`
- `data/metadata/temporal_sequence_metadata.json`
- `data/metadata/temporal_label_mapping.json`

---

## 2. Baseline — Training and Evaluation

Trains the CNN+LSTM+Attention model on the baseline sequences, evaluates
on the test set, saves the trained model and results JSON.

```bat
@echo off
set TF_USE_LEGACY_KERAS=1
"C:\ProgramData\Anaconda3\python.exe" models\train_cnn_lstm_attention.py
```

**Outputs:**
- `models/trained/cnn_lstm_attention.keras`
- `data/metadata/cnn_lstm_attention_results.json`

**Expected results:**
- Test accuracy: ~0.974
- Test macro F1: ~0.975
- Epochs: ~7 (early stopping, patience=3)

---

## 3. Baseline — Inference

Runs prediction on the first sequence in the test split.
Also retrains the model and re-saves it if needed to resolve
Keras version compatibility issues.

```bat
cmd /c "C:\Users\Shinjini\.gemini\antigravity-ide\brain\fba99d60-1489-4638-8981-e91710bb5cd7\scratch\run_inference.bat"
```

Or equivalently (from a cmd.exe session with TF_USE_LEGACY_KERAS already set):

```cmd
set TF_USE_LEGACY_KERAS=1
"C:\ProgramData\Anaconda3\python.exe" run_baseline_inference.py
```

**Output:**
- `data/metadata/baseline_inference_result.json`
- Printed JSON with `prediction`, `confidence`, `probabilities`

**Expected output (sequence index 0, true label = Benign):**
```json
{
  "experiment": "BASELINE (surrogate timestamp-ordered windows from CICFlowMeter CSVs)",
  "sequence_index": 0,
  "true_label": "Benign",
  "prediction": "Benign",
  "confidence": 0.9994,
  "probabilities": {
    "Benign": 0.9994,
    "Brute Force": 0.000059,
    "Botnet": 0.000278,
    "Web Attack": 0.000239
  },
  "correct": true
}
```

> This is BASELINE inference only. Not raw-PCAP inference.

---

## 4. Raw-PCAP — Download Selected PCAP Files

Downloads only the specific UCAP host-capture files needed via HTTP
range requests on the public S3 ZIP archives (no full-archive download).

```cmd
"C:\ProgramData\Anaconda3\python.exe" "C:\Users\Shinjini\.gemini\antigravity-ide\brain\fba99d60-1489-4638-8981-e91710bb5cd7\scratch\download_selected_pcaps.py"
```

**Downloads to:**
`C:\Users\Shinjini\Downloads\extracted_pcap_days\`

Files downloaded:
- `Friday-02-03-2018/UCAP172.31.69.7` (~4.9 MB)
- `Friday-02-03-2018/UCAP172.31.69.15` (~4.9 MB)
- `Friday-02-03-2018/UCAP172.31.69.28` (~5.1 MB)
- `Thursday-22-02-2018/UCAP172.31.69.15` (~1.5 MB)
- `Thursday-22-02-2018/UCAP172.31.69.21` (~35.7 MB)
- `Thursday-22-02-2018/UCAP172.31.69.28` (~25.7 MB)
- `Friday-23-02-2018/UCAP172.31.69.27` (~38.0 MB)
- `Friday-23-02-2018/UCAP172.31.69.28` (~33.6 MB)

Wednesday UCAP25 must be extracted separately from:
`C:\Users\Shinjini\Downloads\pcap_wednesday\pcap.zip`
→ `UCAP172.31.69.25` (~813 MB)

---

## 5. Raw-PCAP — Multi-Day Flow Extraction

Uses a custom high-speed binary PCAP parser (no Scapy) to extract
bidirectional flow records from all downloaded UCAPs.

```cmd
"C:\ProgramData\Anaconda3\python.exe" preprocessing\extract_real_flows_multiday.py
```

**Output:**
- `data/raw/real_flows_multiday.csv` (38,013 flows)

**Expected label counts:**
- Benign: 23,588
- Brute Force: 14,117
- Web Attack: 287
- Botnet: 21

---

## 6. Raw-PCAP — Temporal Grouping and Sequence Builder

Groups flows by normalized bidirectional 5-tuple key, attempts to build
length-10 sequences within each group.

```cmd
"C:\ProgramData\Anaconda3\python.exe" preprocessing\build_real_temporal_sequences.py
```

**Output:**
- `data/processed/real_temporal_sequences/{train,validation,test}.npz`
- `data/processed/real_temporal_sequences/scaler.json`
- `data/metadata/real_temporal_sequence_metadata.json`

**Expected result:**
- 0 valid sequences (maximum group size is 3; sequence length requires 10)

---

## 7. Raw-PCAP — Grouping Validation

Verifies the bidirectional 5-tuple implementation and reports the
group-size distribution.

```cmd
"C:\ProgramData\Anaconda3\python.exe" "C:\Users\Shinjini\.gemini\antigravity-ide\brain\fba99d60-1489-4638-8981-e91710bb5cd7\scratch\verify_grouping.py"
```

**Expected output:**
```
Max group size: 3
Group size distribution:
  Size 1: 37,946
  Size 2-9: 30
  Size >=10: 0
```

---

## Environment

| Package             | Version used |
|---------------------|-------------|
| Python              | 3.11        |
| TensorFlow          | 2.13.0      |
| Keras               | 2.13.1      |
| NumPy               | 1.26.4      |
| Pandas              | (Anaconda default) |
| scikit-learn        | (Anaconda default) |

> `TF_USE_LEGACY_KERAS=1` must be set to prevent TF 2.13 from loading
> the Keras 3 installation in AppData\Roaming\Python\Python311\site-packages.
