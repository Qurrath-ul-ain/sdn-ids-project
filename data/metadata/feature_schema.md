# SDN-IDS ML Feature Schema (Proposed)

Status: **pending controller confirmation**. The controller implementation is not
yet present in this repository (`controller/` currently contains only `.gitkeep`),
so this document records the contract that must be confirmed with Qurrath before
model training and integration.

## Model input

The four-class model should use these five numerical features, in this exact
order:

| Model feature | CICFlowMeter source | Controller source | Required transformation |
| --- | --- | --- | --- |
| `destination_port` | `Dst Port` | destination port | Integer port number. |
| `protocol` | `Protocol` | protocol | IANA IP protocol number (for example, TCP `6`, UDP `17`). |
| `packet_count` | `Tot Fwd Pkts` + `Tot Bwd Pkts` | packet count | Sum the two CIC columns during offline preparation. Controller supplies the total directly. |
| `byte_count` | `TotLen Fwd Pkts` + `TotLen Bwd Pkts` | byte count | Sum the two CIC columns during offline preparation. Controller supplies the total directly. |
| `flow_duration_us` | `Flow Duration` | flow duration | CICFlowMeter uses microseconds. Convert controller duration to microseconds before inference. |

The future preprocessing pipeline must generate these canonical feature names
from the processed CSV, persist their order in `data/metadata/feature_list.json`,
and require the same names in the inference input.

## Excluded fields

- `Timestamp` is excluded because it is collection-time context, not a stable
  real-time flow property.
- Source and destination IP addresses are excluded: they are identifiers and
  would encourage host memorization.
- Source port is excluded because the current processed CICFlowMeter export has
  no source-port column (`Dst Port` is the only port column).
- Directional, packet-size, IAT, rate, TCP-flag, subflow, window, active, and
  idle fields are excluded because the agreed controller interface does not yet
  supply equivalent values.
- `Label` is the target and is never a model feature.

## Controller handoff requirements

Before training, Qurrath should confirm that controller counters represent one
bidirectional flow over the same observation interval as the reported duration.
The controller must provide non-negative numeric values and convert OpenFlow
`duration_sec`/`duration_nsec` to `flow_duration_us`:

```text
flow_duration_us = duration_sec * 1_000_000 + duration_nsec // 1_000
```

The detection layer should keep `source_ip` and `destination_ip` as context for
display and mitigation, but omit them from the ML feature dictionary.

## Dataset impact

`data/processed/four_class_dataset.csv` does **not** need regeneration for this
selection: it already includes all five CIC source columns. The preprocessing
pipeline can derive `packet_count` and `byte_count` when it reads the dataset.
Regenerate only if the label mapping or the selected source columns change.

## Verified dataset inspection

The processed CSV has 80 columns: 79 CICFlowMeter fields plus `Label`. A
streaming inspection of all 4,194,300 rows produced this class distribution:

| Label | Rows |
| --- | ---: |
| Benign | 3,526,232 |
| Brute Force | 380,949 |
| Botnet | 286,191 |
| Web Attack | 928 |

This confirms a severe Web Attack imbalance. Future split and training code
must stratify before fitting preprocessing, and model evaluation must include
per-class precision, recall, and F1 in addition to accuracy.
