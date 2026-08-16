"""Train CNN + LSTM + Attention on the REAL temporal sequences.

This script is the final-stage classifier for the valid flow/session pipeline.
It reuses the same architecture as the baseline surrogate model but is trained
on sequences constructed from proper bidirectional communication group grouping.

Data source: data/processed/real_temporal_sequences/{train,validation,test}.npz
Trained model saved to: models/trained/cnn_lstm_attention_real.keras
Results saved to: data/metadata/real_cnn_lstm_attention_results.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# TensorFlow import -- try user-installed path first
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, r'C:\Users\Shinjini\AppData\Roaming\Python\Python311\site-packages')

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
DATA        = ROOT / 'data' / 'processed' / 'real_temporal_sequences'
META        = ROOT / 'data' / 'metadata'
MODEL_PATH  = ROOT / 'models' / 'trained' / 'cnn_lstm_attention_real.keras'
RESULT_PATH = META / 'real_cnn_lstm_attention_results.json'
SEED        = 42


class Attention(tf.keras.layers.Layer):
    """Soft self-attention over LSTM output sequence."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = tf.keras.layers.Dense(1)

    def call(self, inputs):
        weights = tf.nn.softmax(self.score(inputs), axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray, labels: list[str]) -> dict:
    predictions = probabilities.argmax(axis=1)
    n = len(labels)
    report = classification_report(
        y_true, predictions,
        labels=list(range(n)), target_names=labels,
        output_dict=True, zero_division=0,
    )
    return {
        'accuracy':         float(report['accuracy']),
        'macro_precision':  float(report['macro avg']['precision']),
        'macro_recall':     float(report['macro avg']['recall']),
        'macro_f1':         float(report['macro avg']['f1-score']),
        'weighted_f1':      float(report['weighted avg']['f1-score']),
        'per_class': {
            lbl: {
                'precision': report[lbl]['precision'],
                'recall':    report[lbl]['recall'],
                'f1':        report[lbl]['f1-score'],
                'support':   int(report[lbl]['support']),
            }
            for lbl in labels
        },
        'confusion_matrix': confusion_matrix(
            y_true, predictions, labels=list(range(n))
        ).tolist(),
    }


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)

    # Load metadata
    meta_path = META / 'real_temporal_sequence_metadata.json'
    if not meta_path.exists():
        raise FileNotFoundError(
            f'{meta_path} not found. '
            'Run preprocessing/build_real_temporal_sequences.py first.'
        )
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    label_map   = metadata['label_mapping']           # {name: int}
    labels      = list(label_map.keys())              # ordered label names

    # Load splits
    train_npz = np.load(DATA / 'train.npz')
    val_npz   = np.load(DATA / 'validation.npz')
    test_npz  = np.load(DATA / 'test.npz')

    X_train, y_train = train_npz['X'], train_npz['y']
    X_val,   y_val   = val_npz['X'],   val_npz['y']
    X_test,  y_test  = test_npz['X'],  test_npz['y']

    print(f'Train:      {X_train.shape}, labels unique: {set(y_train.tolist())}')
    print(f'Validation: {X_val.shape},   labels unique: {set(y_val.tolist())}')
    print(f'Test:       {X_test.shape},  labels unique: {set(y_test.tolist())}')

    n_classes = len(labels)

    # Class weights
    counts = np.bincount(y_train, minlength=n_classes)
    counts = np.where(counts == 0, 1, counts)   # avoid division by zero
    class_weights = {
        i: float(len(y_train) / (n_classes * cnt))
        for i, cnt in enumerate(counts)
    }
    print('Class weights:', class_weights)

    # Build model (identical architecture to baseline)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10, 5)),
        tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.LSTM(64, return_sequences=True),
        Attention(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    # Training
    started = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            )
        ],
        verbose=1,
        shuffle=True,
    )
    elapsed = time.perf_counter() - started

    # Evaluate
    val_probs  = model.predict(X_val,  verbose=0)
    test_probs = model.predict(X_test, verbose=0)

    val_result  = compute_metrics(y_val,  val_probs,  labels)
    test_result = compute_metrics(y_test, test_probs, labels)

    print('\n=== VALIDATION RESULTS ===')
    print(json.dumps(val_result, indent=2))
    print('\n=== TEST RESULTS ===')
    print(json.dumps(test_result, indent=2))

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f'\nModel saved to {MODEL_PATH}')

    # Save results
    result = {
        'dataset_type':      'REAL temporal sequences from raw PCAP',
        'source_pcap_files': metadata.get('source_pcap_files', []),
        'architecture': [
            'Conv1D(32,3,same,relu)',
            'MaxPooling1D(2)',
            'LSTM(64,return_sequences=True)',
            'Attention',
            'Dense(32,relu)',
            'Dropout(0.3)',
            f'Dense({n_classes},softmax)',
        ],
        'epochs_completed':    len(history.history['loss']),
        'training_seconds':    elapsed,
        'class_weights':       class_weights,
        'sequence_counts':     metadata['sequence_counts'],
        'sequence_class_counts': metadata['sequence_class_counts'],
        'available_classes':   metadata.get('available_classes', labels),
        'validation':          val_result,
        'test':                test_result,
        'limitation':          metadata.get('limitation', ''),
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'Results saved to {RESULT_PATH}')


if __name__ == '__main__':
    main()
