"""The pooled warden: one judge looking at the text and the image together.

Each "eye" (a text detector, an image detector) produces a raw score. Raw
scores from different detectors are not comparable, so each is converted to an
empirical p-value against *clean* carriers held out for calibration:

    p = fraction of clean carriers whose score is >= this score

A small p means "more suspicious than almost every clean carrier".

The warden flags a (text, image) pair when either channel's p-value is below
alpha/2 (Bonferroni), which keeps the false-alarm rate on clean pairs at about
alpha (5% by default). Its continuous *suspicion* is 1 - min(p_text, p_image),
used for ROC-AUC and as the sender's cost in carrier selection.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclasses.dataclass
class Calibrator:
    clean_scores: np.ndarray

    @classmethod
    def fit(cls, clean_scores: Sequence[float]) -> "Calibrator":
        return cls(np.sort(np.asarray(clean_scores, dtype=float)))

    def p_value(self, scores) -> np.ndarray:
        s = np.atleast_1d(np.asarray(scores, dtype=float))
        n = len(self.clean_scores)
        greater_eq = n - np.searchsorted(self.clean_scores, s, side="left")
        return (greater_eq + 1) / (n + 1)  # +1 smoothing: never exactly 0


@dataclasses.dataclass
class PooledWarden:
    text_cal: Calibrator
    image_cal: Calibrator
    alpha: float = 0.05

    def suspicion(self, text_scores, image_scores) -> np.ndarray:
        pt = self.text_cal.p_value(text_scores)
        pi = self.image_cal.p_value(image_scores)
        return 1.0 - np.minimum(pt, pi)

    def flags(self, text_scores, image_scores) -> np.ndarray:
        pt = self.text_cal.p_value(text_scores)
        pi = self.image_cal.p_value(image_scores)
        return (pt < self.alpha / 2) | (pi < self.alpha / 2)


def auc(clean_scores, stego_scores) -> float:
    y = np.r_[np.zeros(len(clean_scores)), np.ones(len(stego_scores))]
    s = np.r_[clean_scores, stego_scores]
    return float(roc_auc_score(y, s))
