"""Run inference on one real (10,5) sequence from the real temporal dataset.

Loads the trained CNN+LSTM+Attention model and the real test sequences,
selects the first available sequence, and reports:
  - predicted class
  - confidence
  - all class probabilities
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r'C:\Users\Shinjini\AppData\Roaming\Python\Python311\site-packages')

import numpy as np
import tensorflow as tf

ROOT       = Path(__file__).resolve().parents[1]
DATA       = ROOT / 'data' / 'processed' / 'real_temporal_sequences'
META       = ROOT / 'data' / 'metadata'
MODEL_PATH = ROOT / 'models' / 'trained' / 'cnn_lstm_attention_real.keras'


class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = tf.keras.layers.Dense(1)

    def call(self, inputs):
        weights = tf.nn.softmax(self.score(inputs), axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)


def main() -> None:
    # Load label mapping
    meta = json.loads((META / 'real_temporal_sequence_metadata.json').read_text())
    label_map = meta['label_mapping']
    idx_to_label = {v: k for k, v in label_map.items()}

    # Load model
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'Attention': Attention}
    )
    print(f'Model loaded from {MODEL_PATH}')

    # Load test sequences
    test_npz = np.load(DATA / 'test.npz')
    X_test, y_test = test_npz['X'], test_npz['y']
    print(f'Test sequences: {X_test.shape}')

    # Select first sequence
    seq    = X_test[0:1]   # shape (1, 10, 5)
    true_y = int(y_test[0])

    print(f'\nInput sequence shape: {seq.shape}')
    print(f'Input sequence (scaled features):\n{seq[0]}')

    # Run inference
    probs = model.predict(seq, verbose=0)[0]   # shape (n_classes,)

    pred_idx  = int(probs.argmax())
    pred_label = idx_to_label.get(pred_idx, f'class_{pred_idx}')
    confidence = float(probs[pred_idx])

    print(f'\nTrue label:      {idx_to_label.get(true_y, str(true_y))} (index {true_y})')
    print(f'Predicted class: {pred_label} (index {pred_idx})')
    print(f'Confidence:      {confidence:.6f}  ({confidence*100:.2f}%)')
    print('\nClass probabilities:')
    for name, idx in sorted(label_map.items(), key=lambda kv: kv[1]):
        print(f'  {name:<15} : {probs[idx]:.6f}')

    # Save result
    inference_result = {
        'sequence_index':     0,
        'true_label':         idx_to_label.get(true_y, str(true_y)),
        'predicted_class':    pred_label,
        'predicted_index':    pred_idx,
        'confidence':         confidence,
        'class_probabilities': {
            name: float(probs[idx]) for name, idx in label_map.items()
        },
        'sequence_shape':     list(seq.shape[1:]),
        'source':             'real_temporal_sequences/test.npz',
    }
    out_path = META / 'real_inference_result.json'
    out_path.write_text(json.dumps(inference_result, indent=2) + '\n', encoding='utf-8')
    print(f'\nInference result saved to {out_path}')


if __name__ == '__main__':
    main()
