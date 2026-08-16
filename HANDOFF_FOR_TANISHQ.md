# Shinjini → Tanishq: Model Integration Handoff

## What This Provides
A trained 4-class network intrusion detection model that takes a sequence
of 10 network flows and predicts whether it is Benign, Brute Force, Botnet,
or Web Attack traffic.

---

## 1. Trained Model

**File:** `models/trained/cnn_lstm_attention.keras`

Load it like this:

```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # must be before TF import

import tensorflow as tf

class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = tf.keras.layers.Dense(1)
    def call(self, inputs):
        weights = tf.nn.softmax(self.score(inputs), axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)
    def get_config(self):
        return super().get_config()

model = tf.keras.models.load_model(
    "models/trained/cnn_lstm_attention.keras",
    custom_objects={"Attention": Attention}
)
```

> **Environment requirement:** Set `TF_USE_LEGACY_KERAS=1` before starting
> Python. TensorFlow 2.13 must be installed.

---

## 2. Preprocessing / Scaling Pipeline

**Scaler file:** `data/processed/temporal_sequences/scaler.json`

The scaler is a StandardScaler (mean + scale) fit only on training data.
Every input sequence MUST be normalized with this scaler before prediction.

```python
import json
import numpy as np

with open("data/processed/temporal_sequences/scaler.json") as f:
    scaler = json.load(f)

mean  = np.array(scaler["mean"],  dtype=np.float32)   # shape (5,)
scale = np.array(scaler["scale"], dtype=np.float32)   # shape (5,)

def normalize(raw_sequence):
    """raw_sequence: numpy array shape (10, 5) with raw feature values"""
    return (raw_sequence - mean) / scale
```

---

## 3. Required Feature List (exact order)

| Position | Feature name       | Description                              | Units         |
|----------|--------------------|------------------------------------------|---------------|
| 0        | `destination_port` | Destination port of the flow             | integer       |
| 1        | `protocol`         | IP protocol (6=TCP, 17=UDP, etc.)        | integer       |
| 2        | `packet_count`     | Total packets (forward + backward)       | count         |
| 3        | `byte_count`       | Total bytes (forward + backward)         | bytes         |
| 4        | `flow_duration_us` | Duration of the flow                     | microseconds  |

---

## 4. Model Input Format

| Property        | Value         |
|-----------------|---------------|
| Shape           | `(1, 10, 5)`  |
| Dtype           | `float32`     |
| Normalization   | StandardScaler (see scaler.json above) |
| Sequence length | 10 consecutive flows from the same connection group |
| Features        | 5 (in exact order above) |

**How to build the input:**

```python
# raw_flows: list of 10 dicts or rows, each with the 5 features
import numpy as np

raw_sequence = np.array([
    [flow["destination_port"],
     flow["protocol"],
     flow["packet_count"],
     flow["byte_count"],
     flow["flow_duration_us"]]
    for flow in raw_flows   # must be exactly 10 flows, time-sorted
], dtype=np.float32)        # shape (10, 5)

normalized = normalize(raw_sequence)      # shape (10, 5)
model_input = normalized[np.newaxis, ...]  # shape (1, 10, 5)
```

---

## 5. Prediction Output Format

```python
probs = model.predict(model_input, verbose=0)[0]  # shape (4,)

LABELS = ["Benign", "Brute Force", "Botnet", "Web Attack"]

result = {
    "prediction":    LABELS[int(np.argmax(probs))],
    "confidence":    float(np.max(probs)),
    "probabilities": {lbl: float(probs[i]) for i, lbl in enumerate(LABELS)}
}
```

**Example output:**
```json
{
  "prediction":   "Benign",
  "confidence":   0.9994,
  "probabilities": {
    "Benign":      0.9994,
    "Brute Force": 0.000059,
    "Botnet":      0.000278,
    "Web Attack":  0.000239
  }
}
```

---

## 6. Label Mapping

| Integer | Label       |
|---------|-------------|
| 0       | Benign      |
| 1       | Brute Force |
| 2       | Botnet      |
| 3       | Web Attack  |

File: `data/metadata/temporal_label_mapping.json`

---

## 7. Complete End-to-End Usage Example

```python
import os, json
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

# ── Load scaler ──────────────────────────────────────────────────────────────
with open("data/processed/temporal_sequences/scaler.json") as f:
    sc = json.load(f)
mean  = np.array(sc["mean"],  dtype=np.float32)
scale = np.array(sc["scale"], dtype=np.float32)

# ── Load model ───────────────────────────────────────────────────────────────
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = tf.keras.layers.Dense(1)
    def call(self, inputs):
        weights = tf.nn.softmax(self.score(inputs), axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)
    def get_config(self):
        return super().get_config()

model = tf.keras.models.load_model(
    "models/trained/cnn_lstm_attention.keras",
    custom_objects={"Attention": Attention}
)

LABELS = ["Benign", "Brute Force", "Botnet", "Web Attack"]

# ── Predict ──────────────────────────────────────────────────────────────────
def predict_sequence(raw_flows_10x5: np.ndarray) -> dict:
    """
    raw_flows_10x5: numpy array shape (10, 5) with RAW (unnormalized) values.
    Feature order: [destination_port, protocol, packet_count, byte_count, flow_duration_us]
    Returns dict with prediction, confidence, probabilities.
    """
    assert raw_flows_10x5.shape == (10, 5), "Input must be shape (10, 5)"
    normalized = (raw_flows_10x5.astype(np.float32) - mean) / scale
    probs = model.predict(normalized[np.newaxis, ...], verbose=0)[0]
    return {
        "prediction":    LABELS[int(np.argmax(probs))],
        "confidence":    float(np.max(probs)),
        "probabilities": {lbl: float(probs[i]) for i, lbl in enumerate(LABELS)}
    }
```

---

## 8. Files to Hand Over to Tanishq

Share the entire project folder, or at minimum these files:

```
sdn-ids-project/
├── models/
│   └── trained/
│       └── cnn_lstm_attention.keras       ← trained model weights
├── data/
│   ├── processed/
│   │   └── temporal_sequences/
│   │       └── scaler.json               ← normalization parameters
│   └── metadata/
│       ├── temporal_label_mapping.json   ← {label: integer} mapping
│       └── baseline_inference_result.json ← verified example output
├── run_baseline_inference.py              ← working inference script
└── REPRODUCIBILITY.md                    ← commands to reproduce everything
```

**Model performance (test set):**
- Accuracy: 97.4 %
- Macro F1: 97.5 %
- Brute Force: 100 % precision/recall
- Web Attack: 100 % precision/recall
- Botnet: 95.1 % F1
- Benign: 94.9 % F1

> Note: These results are from the baseline surrogate-window experiment
> on the CSE-CIC-IDS2018 dataset. See FINAL_REPORT.md for full methodology.
