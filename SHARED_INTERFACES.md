# Shared Interfaces

## Flow Statistics

The controller will provide network flow information to the IDS.

Initial fields:

- source IP
- destination IP
- source port
- destination port
- protocol
- packet count
- byte count
- flow duration

## ML Input

The ML model will receive numerical network-flow features after preprocessing.

The exact feature list will be finalized before ML integration.

## Detection Output

The IDS should produce:

- source IP
- destination IP
- prediction
- attack type
- confidence score

Example:

{
  "source_ip": "10.0.0.5",
  "destination_ip": "10.0.0.2",
  "prediction": "Malicious",
  "attack_type": "Brute Force",
  "confidence": 0.94
}


**QURRATH WORK**
## SDN Infrastructure — Stage 2–4 Status

### Stage 2 — Mininet Topology
- Healthcare SDN topology verified successfully.
- Hosts: h1, h2, h3, h4.
- Switch: s1.
- Ryu controller: 127.0.0.1:6653.
- OpenFlow version: OpenFlow 1.3.

### Stage 3 — Ryu Controller Verification
- Ryu successfully connected to switch s1.
- `pingall`: **0% dropped (12/12 received)**.
- `h1 -> h4`: **0% packet loss (3/3 received)**.
- OpenFlow traffic forwarding verified.

### Stage 4 — OpenFlow Verification
- Flow table verified using `ovs-ofctl`.
- Default `priority=0` controller rule confirmed.
- Dynamic `priority=1` forwarding rules confirmed.
- Flow packet/byte counters verified.
- Switch port packet/byte statistics verified.

### Status
- **Stage 2 — COMPLETE**
- **Stage 3 — COMPLETE**
- **Stage 4 — COMPLETE**
- **Stage 11 — PENDING**
  
## SHINJINI WORK
## ML-Based Combined Attack Detection — Stage 5–10 Status

### Stage 5 — ML Data Preprocessing
- CSE-CIC-IDS2018 network-flow data prepared for the combined IDS.
- Invalid/header-like records removed.
- Attack labels normalized into four classes:
  - Benign
  - Brute Force
  - Botnet
  - Web Attack
- Required ML features finalized:
  - `destination_port`
  - `protocol`
  - `packet_count`
  - `byte_count`
  - `flow_duration_us`
- Numerical preprocessing and training-data-based scaling implemented.
- Original CICFlowMeter CSV files preserved without modification.

### Stage 6 — Temporal Dataset and Sequence Construction
- Required input sequence length finalized as **10**.
- Model input shape finalized as **(10,5)**.
- Temporal ordering and communication-group methodology implemented.
- Raw PCAP flow metadata preserved, including IP addresses, ports, protocol and timestamps.
- Normalized bidirectional communication grouping implemented using:
  `{Src IP, Src Port} ↔ {Dst IP, Dst Port} + Protocol`.
- Leakage prevention implemented by partitioning communication groups before constructing overlapping sequences.

### Stage 7 — CNN–LSTM–Attention Model
- Unified four-class CNN–LSTM–Attention model implemented.
- Architecture:
  - Conv1D(32)
  - MaxPooling1D
  - LSTM(64)
  - Attention
  - Dense(32)
  - Dropout
  - Softmax(4)
- Input shape: **(10,5)**.
- Output classes:
  - Benign
  - Brute Force
  - Botnet
  - Web Attack.
- Random Forest and DNN are not part of the final model.

### Stage 8 — Model Training and Evaluation
- Baseline CNN–LSTM–Attention training completed using the prepared surrogate temporal dataset.
- Baseline dataset:
  - 2,000 cleaned flow records.
  - 1,919 surrogate sequences.
  - 500 samples per class.
- Baseline results:
  - Test Accuracy: **96.70%**
  - Macro F1: **96.78%**
  - Weighted F1: **96.71%**
- Training artifacts and model files saved.

### Stage 9 — Raw-PCAP Validation
- Raw CSE-CIC-IDS2018 PCAP flow extraction pipeline implemented and executed.
- **38,013 real flow records** extracted from selected PCAP sources.
- Flow/session metadata preserved where available.
- Required temporal grouping method verified.
- Maximum communication-group size observed: **3 flows**.
- Required sequence length: **10 flows**.
- Therefore, no valid raw-PCAP `(10,5)` sequences could be constructed under the prescribed grouping rule.
- No artificial session IDs or fabricated temporal groups were introduced.
- No final raw-PCAP performance metrics were fabricated.

### Stage 10 — IDS Inference Interface
- ML inference interface defined for controller-provided flow statistics.
- Five numerical features are extracted and validated before inference.
- The trained CNN–LSTM–Attention model produces four-class probabilities.
- Detection output includes:
  - source IP
  - destination IP
  - prediction
  - attack type
  - confidence score.
- Example output:
  ```json
  {
    "source_ip": "10.0.0.5",
    "destination_ip": "10.0.0.2",
    "prediction": "Malicious",
    "attack_type": "Brute Force",
    "confidence": 0.94
  }
````

### Status

* **Stage 5 — COMPLETE**
* **Stage 6 — COMPLETE / RAW-PCAP SEQUENCE LIMITATION IDENTIFIED**
* **Stage 7 — COMPLETE**
* **Stage 8 — COMPLETE FOR BASELINE**
* **Stage 9 — COMPLETE / RAW-PCAP SEQUENCE BLOCKER DOCUMENTED**
* **Stage 10 — COMPLETE**





