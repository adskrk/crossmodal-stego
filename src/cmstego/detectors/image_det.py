"""Image steganalysis used as the warden's "image eye".

* ``spa_score`` — Sample Pairs Analysis (Dumitrescu, Wu & Wang, 2003), a
  training-free structural attack that *estimates the LSB-replacement payload*.
  Re-implemented from Aletheia's version (MIT). It targets LSB replacement only.
* ``EffNetDetector`` — Aletheia's pretrained EfficientNet-B0 detectors (MIT),
  trained by its author on ALASKA2 colour images for LSBR, LSBM or HILL at
  512x512. We run them unchanged on our 256x256 crops (the network ends in
  global average pooling, so input size can change); that is itself a
  mismatch versus their training setup and is reported as such.

The detectors return a score in [0, 1]: higher = "more likely stego".
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_MODEL_DIR = os.environ.get("ALETHEIA_MODELS", os.path.join(ROOT, "vendor", "aletheia", "aletheia-models"))


def spa_channel(I: np.ndarray) -> float:
    I = I.astype(np.int64)
    r, s = I[:-1, :], I[1:, :]
    lsb0 = (s & 1) == 0
    lsb1 = ~lsb0
    x = np.sum((lsb0 & (r < s)) | (lsb1 & (r > s)))
    y = np.sum((lsb0 & (r > s)) | (lsb1 & (r < s)))
    msb = I & 0xFE
    k = np.sum(msb[:-1, :] == msb[1:, :])
    if k == 0:
        return 0.0
    a, b, c = 2 * k, 2 * (2 * x - r.size), y - x
    disc = complex(b * b - 4 * a * c) ** 0.5
    bp, bm = (-b + disc) / (2 * a), (-b - disc) / (2 * a)
    beta = min(bp.real, bm.real)
    return float(2 * beta)


def spa_score(img: np.ndarray) -> float:
    """Mean SPA payload estimate over R, G, B, clipped to [0, 1]."""
    est = [spa_channel(img[..., k]) for k in range(3)]
    return float(np.clip(np.mean(est), 0.0, 1.0))


def spam_features(img: np.ndarray, T: int = 3) -> np.ndarray:
    """SPAM-style features (after Pevný, Bas & Fridrich, 2010), simplified.

    For each colour channel, take neighbouring-pixel differences in the
    horizontal and vertical directions, clip them to [-T, T], and build the
    first-order Markov transition matrix P(d_{i+1} | d_i). Matrices for
    opposite directions are averaged. ±1 embedding adds noise to these
    differences, which shows up as changed transition probabilities.
    Output: 3 channels x 2 directions x (2T+1)^2 = 294 features.
    """
    feats = []
    k = 2 * T + 1
    for c in range(3):
        x = img[..., c].astype(np.int64)
        for axis in (0, 1):
            mats = []
            for sign in (1, -1):
                d = np.diff(x, axis=axis) * sign
                d = np.clip(d, -T, T) + T
                a = d.take(range(d.shape[axis] - 1), axis=axis).ravel()
                b = d.take(range(1, d.shape[axis]), axis=axis).ravel()
                m = np.zeros((k, k))
                np.add.at(m, (a, b), 1)
                m /= np.maximum(m.sum(1, keepdims=True), 1)
                mats.append(m)
            feats.append(((mats[0] + mats[1]) / 2).ravel())
    return np.concatenate(feats)


class EffNetDetector:
    """Aletheia pretrained EfficientNet-B0 ('A' models: cover vs stego)."""

    def __init__(self, method: str, size: int = 256, model_dir: str = DEFAULT_MODEL_DIR):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        import efficientnet.tfkeras as efn
        from tensorflow.keras import layers as L
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("float32")
        name = {"lsbr": "lsbr", "lsbm": "lsbm", "hill": "hill"}[method]
        path = os.path.join(model_dir, f"effnetb0-A-alaska2-{name}.h5")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; clone Aletheia's aletheia-models folder")
        base = efn.EfficientNetB0(input_shape=(size, size, 3), weights=None, include_top=False)
        self.model = tf.keras.Sequential([base, L.GlobalAveragePooling2D(), L.Dense(2, activation="softmax", dtype="float32")])
        self.model.load_weights(path)
        self.method = method
        self.size = size

    def score(self, images: Sequence[np.ndarray], batch: int = 16, chunk: int = 64) -> np.ndarray:
        # Convert to float32 one chunk at a time: stacking thousands of images at
        # once ran the experiment out of memory.
        out = []
        for i in range(0, len(images), chunk):
            X = np.stack([im[: self.size, : self.size, :3] for im in images[i : i + chunk]]).astype("float32") / 255.0
            out.append(self.model.predict(X, batch_size=batch, verbose=0)[:, 1])  # column 1 = stego (Aletheia's label order)
        return np.concatenate(out) if out else np.zeros(0)
