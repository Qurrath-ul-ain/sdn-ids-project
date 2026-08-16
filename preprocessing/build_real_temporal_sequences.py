"""Build real temporal sequences from PCAP-derived flow records.

Pipeline
--------
1. Load wednesday_flows_raw.csv produced by extract_flows_from_pcap.py
2. Build a normalized bidirectional communication key per flow
3. Assign complete communication groups to train/validation/test (no leakage)
4. Fit scaler only on training data
5. Construct overlapping (10,5) windows inside each partition
6. Save sequences, scaler, metadata

Grouping key (bidirectional, canonical ordering)
-------------------------------------------------
    group_id = (min(src_ip, dst_ip), src_port_of_min, max(src_ip, dst_ip), dst_port_of_max, protocol)

This is DIFFERENT from the surrogate dataset which had NO grouping.

Sequence features (in order)
-----------------------------
    destination_port
    protocol
    packet_count       (tot_fwd_pkts + tot_bwd_pkts)
    byte_count         (totlen_fwd_bytes + totlen_bwd_bytes)
    flow_duration_us
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[1]
IN_CSV   = ROOT / 'data' / 'raw' / 'real_flows_multiday.csv'
OUT_DIR  = ROOT / 'data' / 'processed' / 'real_temporal_sequences'
META_DIR = ROOT / 'data' / 'metadata'

SEQUENCE_LENGTH = 10
STRIDE          = 1
SEED            = 42

FEATURE_NAMES = [
    'destination_port',
    'protocol',
    'packet_count',
    'byte_count',
    'flow_duration_us',
]

LABEL_MAP   = {'Benign': 0, 'Brute Force': 1, 'Botnet': 2, 'Web Attack': 3}
LABEL_NAMES = list(LABEL_MAP)

# Split fractions
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# test = remainder


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def make_group_key(row: pd.Series) -> str:
    """Normalized bidirectional communication group identifier."""
    ip_a, port_a = str(row['src_ip']),  int(row['src_port'])
    ip_b, port_b = str(row['dst_ip']),  int(row['dst_port'])
    proto        = int(row['protocol'])
    if (ip_a, port_a) <= (ip_b, port_b):
        return f'{ip_a}-{port_a}-{ip_b}-{port_b}-{proto}'
    return f'{ip_b}-{port_b}-{ip_a}-{port_a}-{proto}'


def build_sequences(partition_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding windows inside a single partition, grouped by comm key."""
    Xs, ys = [], []
    # Process each communication group separately (no cross-group windows)
    for group_id, group_df in partition_df.groupby('group_id'):
        group_sorted = group_df.sort_values('timestamp')
        values  = group_sorted[FEATURE_NAMES].to_numpy(dtype=np.float32)
        targets = group_sorted['label_int'].to_numpy(dtype=np.int64)
        if len(values) < SEQUENCE_LENGTH:
            continue
        for start in range(0, len(values) - SEQUENCE_LENGTH + 1, STRIDE):
            Xs.append(values[start:start + SEQUENCE_LENGTH])
            ys.append(int(targets[start + SEQUENCE_LENGTH - 1]))
    if Xs:
        return np.stack(Xs, axis=0), np.array(ys, dtype=np.int64)
    return np.empty((0, SEQUENCE_LENGTH, 5), dtype=np.float32), np.empty(0, dtype=np.int64)


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(
            f'Flow CSV not found: {IN_CSV}\n'
            'Run preprocessing/extract_real_flows_multiday.py first.'
        )

    print(f'Loading {IN_CSV} ...', flush=True)
    df = pd.read_csv(IN_CSV, low_memory=False)
    print(f'  {len(df):,} flow records loaded', flush=True)

    # -----------------------------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------------------------
    df['destination_port'] = pd.to_numeric(df['dst_port'],          errors='coerce')
    df['protocol']         = pd.to_numeric(df['protocol'],          errors='coerce')
    df['packet_count']     = pd.to_numeric(df['packet_count'],      errors='coerce')
    df['byte_count']       = pd.to_numeric(df['byte_count'],        errors='coerce')
    df['flow_duration_us'] = pd.to_numeric(df['flow_duration_us'],  errors='coerce')

    # Drop rows with NaN in required fields
    required = FEATURE_NAMES + ['timestamp', 'label', 'src_ip', 'src_port', 'dst_ip', 'dst_port']

    df = df.dropna(subset=required).reset_index(drop=True)

    # Map labels
    df['label_canonical'] = df['label'].astype(str).str.strip()
    df['label_int'] = df['label_canonical'].map(LABEL_MAP)
    df = df.dropna(subset=['label_int']).reset_index(drop=True)
    df['label_int'] = df['label_int'].astype(int)

    print(f'  {len(df):,} rows after cleaning', flush=True)
    print('  Label distribution:', df['label_canonical'].value_counts().to_dict(), flush=True)

    # -----------------------------------------------------------------------
    # Bidirectional communication group key
    # -----------------------------------------------------------------------
    df['group_id'] = df.apply(make_group_key, axis=1)
    n_groups = df['group_id'].nunique()
    print(f'  {n_groups:,} unique communication groups', flush=True)

    # -----------------------------------------------------------------------
    # Group-level split (assign entire groups to partitions)
    # Stratify by the majority label of each group
    # -----------------------------------------------------------------------
    # Get per-group majority label and size
    group_info = df.groupby('group_id').agg(
        majority_label=('label_int', lambda x: x.mode()[0]),
        n_flows=('flow_id', 'count'),
    ).reset_index()

    rng = np.random.default_rng(SEED)
    train_groups, val_groups, test_groups = set(), set(), set()

    for label_int in range(len(LABEL_NAMES)):
        subset = group_info[group_info['majority_label'] == label_int]['group_id'].to_numpy()
        if len(subset) == 0:
            continue
        rng.shuffle(subset)
        n_train = max(1, int(len(subset) * TRAIN_FRAC))
        n_val   = max(1, int(len(subset) * VAL_FRAC))
        train_groups.update(subset[:n_train])
        val_groups.update(subset[n_train:n_train + n_val])
        test_groups.update(subset[n_train + n_val:])

    # Assign split to rows
    split_map = (
        {g: 'train' for g in train_groups} |
        {g: 'validation' for g in val_groups} |
        {g: 'test' for g in test_groups}
    )
    df['split'] = df['group_id'].map(split_map)
    df = df.dropna(subset=['split'])

    print('\nGroup counts per split:')
    for split in ('train', 'validation', 'test'):
        n = df[df['split'] == split]['group_id'].nunique()
        print(f'  {split}: {n} groups, {(df["split"] == split).sum():,} flows')

    # -----------------------------------------------------------------------
    # Scaler: fit on training data only
    # -----------------------------------------------------------------------
    train_df = df[df['split'] == 'train']
    mean  = train_df[FEATURE_NAMES].mean().to_numpy()
    scale = train_df[FEATURE_NAMES].std().to_numpy()
    scale[scale == 0] = 1.0  # avoid division by zero

    for feat, m, s in zip(FEATURE_NAMES, mean, scale):
        df[feat] = (df[feat] - m) / s

    # -----------------------------------------------------------------------
    # Build sequences per partition
    # -----------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seq_counts = {}
    seq_class_counts = {}

    for split in ('train', 'validation', 'test'):
        part_df = df[df['split'] == split].copy()
        X, y    = build_sequences(part_df)
        np.savez_compressed(OUT_DIR / f'{split}.npz', X=X.astype(np.float32), y=y)
        seq_counts[split] = int(len(y))
        seq_class_counts[split] = {
            name: int((y == idx).sum()) for name, idx in LABEL_MAP.items()
        }
        print(f'\n{split}: {len(y):,} sequences; shape {X.shape}; class dist {seq_class_counts[split]}')

    # Scaler
    scaler_data = {
        'feature_order': FEATURE_NAMES,
        'mean':          mean.tolist(),
        'scale':         scale.tolist(),
    }
    write_json(OUT_DIR / 'scaler.json', scaler_data)

    # Metadata
    present_labels = [n for n in LABEL_NAMES if df['label_canonical'].isin([n]).any()]
    metadata = {
        'source_csv':              str(IN_CSV),
        'source_type':             'REAL bidirectional flow records from raw PCAP',
        'temporal_window_type':    'sliding windows constructed WITHIN each communication group',
        'grouping_key':            '(min(src_ip,src_port), max(src_ip,src_port), protocol) -- normalized bidirectional',
        'sequence_length':         SEQUENCE_LENGTH,
        'stride':                  STRIDE,
        'feature_order':           FEATURE_NAMES,
        'input_shape':             [SEQUENCE_LENGTH, len(FEATURE_NAMES)],
        'label_mapping':           LABEL_MAP,
        'total_flow_records':      int(len(df)),
        'total_groups':            int(df['group_id'].nunique()),
        'label_distribution':      df['label_canonical'].value_counts().to_dict(),
        'sequence_counts':         seq_counts,
        'sequence_class_counts':   seq_class_counts,
        'split_method':            (
            'Group-level stratified split: complete communication groups '
            'assigned to partitions before window construction. '
            'No group appears in more than one partition.'
        ),
        'scaler':                  'StandardScaler fit on training flows only',
        'available_classes':       present_labels,
        'limitation': (
            'Wednesday PCAP (4 small UCAP files) contains only Benign + Brute Force (SSH). '
            'Botnet and Web Attack classes require Friday-02-03 and Friday/Thursday-02/22-02 PCAPs. '
            'All four classes are required for the final production model.'
        ),
    }
    write_json(META_DIR / 'real_temporal_sequence_metadata.json', metadata)
    print('\nMetadata saved.')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
