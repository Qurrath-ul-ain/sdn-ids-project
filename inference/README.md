# SDN-IDS Inference Interface

`HybridIDS` loads the existing `random_forest.joblib` and `dnn.keras` files.
It never retrains either model. The input must contain exactly the five keys
listed in `data/metadata/feature_list.json`:

```python
from inference import HybridIDS

ids = HybridIDS()
flow = {
    "destination_port": 443,
    "protocol": 6,
    "packet_count": 16,
    "byte_count": 4326,
    "flow_duration_us": 141385,
}

result = ids.predict(flow)                  # final hybrid: RF 0.6 + DNN 0.4
rf_result = ids.predict(flow, model="random_forest")  # RF-only option
```

The method validates exact keys, converts values to finite numeric values, and
applies the training-fitted standard scaler from `models/preprocessing/scaler.json`.
It returns `prediction`, `confidence`, and probabilities in this order:

```text
Benign, Brute Force, Botnet, Web Attack
```

IP addresses and other context fields must be carried outside the ML feature
dictionary. Controller compatibility remains pending Qurrath confirmation.
