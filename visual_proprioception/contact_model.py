"""
Load an exported laser contact model and score frames.

Uses only numpy and OpenCV (cv2.dnn for ONNX), so the ROS node runs without
PyTorch. contact_train.py writes the model files:

    model.json   kind ('spot' | 'image' | 'combined' | 'temporal'), feature
                 normalisation, thresholds, smoothing, and the spot/window
                 settings used in training
    model.onnx   image, combined: inputs 'image' (1x3xSxS, ImageNet-normalised
                 RGB) and, for 'combined', 'spot' (1 x len(FEATURE_NAMES), raw
                 features); output 'prob'
                 temporal: input 'image', output 'embedding'; the small head
                 that compares it with a past frame's embedding is in model.json

Models trained with --target depth output how many mm the fiber is short of the
cutting depth instead of P(contact) (target == 'depth'); their thresholds are in mm
and a frame counts as "at depth" when the output is at or below them.

The temporal model keeps the embeddings of the frames it scored in the last
lag['max'] s, so pass each frame's timestamp to predict() and call reset() when
the stream restarts.
"""

from collections import deque
import json
import math
import os

import cv2
import numpy as np

from .laser_spot_features import (crop_square, crop_to_tensor, FEATURE_NAMES, feature_vector,
                                  SPOT_PARAMS, WINDOW_PARAMS)

DEFAULT_THRESHOLDS = {'suspect': 0.6, 'confirm': 0.8}


class ContactModel:
    """Contact probability from the current frame and its SpotHistory features."""

    def __init__(self, path):
        path = os.path.expanduser(path)
        with open(path) as f:
            self.meta = json.load(f)
        if tuple(self.meta['feature_names']) != FEATURE_NAMES:
            raise ValueError(f'{path} was trained with features {self.meta["feature_names"]}, '
                             f'this code computes {list(FEATURE_NAMES)}')
        self.kind = self.meta['kind']
        self.target = self.meta.get('target', 'label')
        self.thresholds = {**DEFAULT_THRESHOLDS, **self.meta.get('thresholds', {})}
        self.spot_params = {**SPOT_PARAMS, **self.meta.get('spot_params', {})}
        self.window = {**WINDOW_PARAMS, **self.meta.get('window', {})}
        if self.kind == 'spot':
            self.mean = np.array(self.meta['mean'], np.float32)
            self.std = np.array(self.meta['std'], np.float32)
            self.weights = np.array(self.meta['weights'], np.float32)
            self.bias = float(self.meta['bias'])
        elif self.kind in ('image', 'combined', 'temporal'):
            onnx_path = os.path.join(os.path.dirname(path), self.meta['onnx'])
            self.net = cv2.dnn.readNetFromONNX(onnx_path)
            self.crop_size = int(self.meta['crop_size'])
            self.input_size = int(self.meta['input_size'])
        else:
            raise ValueError(f'Unknown model kind {self.kind!r} in {path}')
        if self.kind == 'temporal':
            self.head = {k: np.asarray(v, np.float32) for k, v in self.meta['head'].items()}
            self.lag = self.meta['lag']
        self.smooth_sec = float(self.meta.get('smooth_sec') or 0.0)
        self.recent = deque()   # (t, embedding) of the frames scored in the last lag['max'] s

    def reset(self):
        """Forget past frames (call when the image stream restarts)."""
        self.recent.clear()

    def _past_embedding(self, t):
        """Embedding of the frame closest to t - lag['target'] within the lag window."""
        while self.recent and (self.recent[0][0] < t - self.lag['max'] or self.recent[-1][0] > t):
            if self.recent[-1][0] > t:      # time went backwards (bag restart): start over
                self.recent.clear()
            else:
                self.recent.popleft()
        best = None
        for tp, emb in self.recent:
            if tp <= t - self.lag['min'] and (
                    best is None or abs(tp - (t - self.lag['target'])) < best[0]):
                best = (abs(tp - (t - self.lag['target'])), emb)
        return None if best is None else best[1]

    def predict(self, image, feats, center, t=None):
        """
        Return P(contact) (or mm to the cutting depth) for a BGR frame and its context.

        t is only needed by the temporal model; without it no past frame is used.
        """
        x = feature_vector(feats)
        if self.kind == 'spot':
            z = float(((x - self.mean) / self.std) @ self.weights + self.bias)
            return self._output(z)
        crop = crop_square(image, center, self.crop_size)
        self.net.setInput(crop_to_tensor(crop, self.input_size), 'image')
        if self.kind == 'combined':
            self.net.setInput(x[None], 'spot')
        out = np.asarray(self.net.forward(), np.float32).ravel()
        if self.kind != 'temporal':
            return float(out[0])
        past = self._past_embedding(t) if t is not None else None
        if t is not None:
            self.recent.append((t, out.copy()))
        h = self.head
        spot = np.maximum(((x - h['spot_mean']) / h['spot_std']) @ h['spot_weight'].T
                          + h['spot_bias'], 0.0)
        z = float(np.concatenate([out, out - (out if past is None else past), spot])
                  @ h['out_weight'] + h['out_bias'])
        return self._output(z)

    def _output(self, z):
        """Probability for contact models; mm short of the cutting depth for depth models."""
        return z if self.target == 'depth' else 1.0 / (1.0 + math.exp(-z))

    def at_threshold(self, value, threshold):
        """Return whether value (from predict) meets threshold: P >= t, or depth <= t mm."""
        return value <= threshold if self.target == 'depth' else value >= threshold


class CausalMean:
    """Running mean of the values from the last window_s seconds (window_s <= 0: pass-through)."""

    def __init__(self, window_s):
        self.window_s = float(window_s)
        self.values = deque()

    def reset(self):
        self.values.clear()

    def update(self, t, value):
        if self.window_s <= 0:
            return value
        if self.values and self.values[-1][0] > t:
            self.values.clear()
        self.values.append((t, value))
        while self.values[0][0] <= t - self.window_s:
            self.values.popleft()
        return sum(v for _, v in self.values) / len(self.values)
