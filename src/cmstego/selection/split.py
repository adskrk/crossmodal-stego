"""Choosing the split: how much of the lock-box package goes into the text.

With the all-or-nothing package (aont.py) any split is equally safe for
confidentiality, so the split can be chosen for *detectability* only. More
package bytes in the text mean a longer stego-text (more evidence for a text
detector); fewer bytes in the image mean a lower embedding rate (less evidence
for an image detector). Whether a learned policy beats a fixed rule is an
empirical question, answered here on cached measurements.

Inputs (all measured by exp_text.py / exp_image.py, nothing invented):
  * the image tables from carrier.build_tables (HILL detector scores per cover
    and payload rate, interpolated between measured rates);
  * text held-out scores for several text-fragment sizes at C = 0, one results
    folder per size (e.g. results_gpt2 for 24 bytes, results_gpt2/len64,
    results_gpt2/len160), each scored by the same detectors.

Per message the sender picks a cover (of K), a prompt (of M) and a text size
(of the measured sizes). The text size includes the 10-byte fragment header,
so it carries ``size - 10`` package bytes; the image carries the rest.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..prompts import PROMPTS
from ..warden import Calibrator
from . import carrier as P

FRAGMENT_HEADER = 10
PACKAGE_OVERHEAD = 19  # aont: 16-byte key + 3-byte header (before compression savings)
SPLIT_MESSAGE_BYTES = [64, 256, 1024, 4096]


@dataclasses.dataclass
class SplitTables:
    base: P.Tables
    sizes: List[int]  # text fragment sizes (bytes), ascending
    text: Dict[int, pd.DataFrame]  # size -> held-out stego scores (prompt_id, j, score_<det>)
    text_cal: Dict[Tuple[str, int], Calibrator]  # (text detector, size)
    tokens: Dict[int, float]  # size -> mean tokens of the stego-texts

    def text_p(self, w: str, size: int, row: pd.Series) -> float:
        det = P.WARDENS[w][1]
        return float(self.text_cal[(det, size)].p_value(row[f"score_{det}"])[0])

    @property
    def wardens(self) -> Tuple[str, ...]:
        return tuple(w for w in self.base.wardens
                     if all(f"score_{P.WARDENS[w][1]}" in self.text[s] for s in self.sizes))


def build_split_tables(base: P.Tables, size_dirs: Dict[int, str], text_tag: str = "text_filter_") -> SplitTables:
    text, cal, toks = {}, {}, {}
    pid = {p: i for i, p in enumerate(PROMPTS)}
    for size, d in sorted(size_dirs.items()):
        held = pd.read_csv(os.path.join(d, text_tag + "heldout_scores.csv"))
        extra = os.path.join(d, text_tag + "heldout_scores_style.csv")
        if os.path.exists(extra):
            held = pd.concat([held, pd.read_csv(extra)], ignore_index=True)
        held = held[np.isclose(held["C"], 0.0)].copy()
        held["prompt_id"] = held["prompt"].map(pid)
        by = {det: g.set_index(["prompt_id", "j"]) for det, g in held.groupby("detector")}
        text[size] = pd.DataFrame({f"score_{det}": g["stego_score"] for det, g in by.items()}).reset_index()
        for det, g in by.items():
            cal[(det, size)] = Calibrator.fit(g["cover_score"].values)
        meas = pd.read_csv(os.path.join(d, text_tag + "measurements.csv"))
        toks[size] = float(meas[np.isclose(meas["C"], 0.0)]["tokens"].mean())
    return SplitTables(base, sorted(size_dirs), text, cal, toks)


@dataclasses.dataclass
class SplitEpisode:
    L: int  # secret bytes
    covers: np.ndarray
    prompts: np.ndarray
    pick: Dict[Tuple[int, int], int]  # (prompt, size) -> row in SplitTables.text[size]


def make_episode(S: SplitTables, rng, cover_split: str, prompt_ids: List[int], K: int = 8, M: int = 4) -> SplitEpisode:
    pool = np.flatnonzero(S.base.covers["split"].values == cover_split)
    covers = rng.choice(pool, size=K, replace=False)
    prompts = rng.choice(prompt_ids, size=M, replace=False)
    pick = {}
    for p in prompts:
        for size in S.sizes:
            rows = np.flatnonzero(S.text[size]["prompt_id"].values == p)
            pick[(int(p), size)] = int(rng.choice(rows))
    return SplitEpisode(int(rng.choice(SPLIT_MESSAGE_BYTES)), covers, prompts, pick)


def image_bytes(L: int, size: int) -> int:
    return max(L + PACKAGE_OVERHEAD - (size - FRAGMENT_HEADER), 16)


def evaluate(S: SplitTables, ep: SplitEpisode, k: int, m: int, s: int, w: str) -> Tuple[float, bool]:
    """Pooled suspicion (1 - min p) and whether warden w flags; "ENS" = worst case over P.ENSEMBLE."""
    if w == "ENS":
        res = [evaluate(S, ep, k, m, s, v) for v in P.ENSEMBLE]
        return max(r[0] for r in res), any(r[1] for r in res)
    size = S.sizes[s]
    row = S.text[size].iloc[ep.pick[(int(ep.prompts[m]), size)]]
    pt = S.text_p(w, size, row)
    ci = int(ep.covers[k])
    rate = (image_bytes(ep.L, size) + P.IMAGE_OVERHEAD_BYTES) * 8 / P.COVER_VALUES
    if not S.base.image_feasible(ci, rate):
        return 1.0, True
    pi = S.base.image_p(w, ci, rate)
    return 1.0 - min(pt, pi), (pt < P.ALPHA / 2) or (pi < P.ALPHA / 2)


def observation(S: SplitTables, ep: SplitEpisode) -> np.ndarray:
    T = S.base
    cf = T.covers.loc[ep.covers, ["log_texture", "mean", "std", "saturated"]].values - np.array([3.0, 0.5, 0.25, 0.0])
    cap = T.covers.loc[ep.covers, "capacity_bits"].values / P.COVER_VALUES
    cf = np.c_[cf, cap * 2]
    pf = T.prompt_feats[ep.prompts]
    return np.concatenate([cf.ravel(), pf.ravel(), [np.log2(ep.L) / 12]]).astype(np.float32)


def choice_from_scores(S: SplitTables, ep: SplitEpisode, a) -> Tuple[int, int, int]:
    """Scoring policy: weights over cover and prompt features, plus one size knob."""
    a = np.asarray(a, dtype=float)
    obs = observation(S, ep)
    k, m = len(ep.covers), len(ep.prompts)
    cf = obs[: k * P.COVER_FEATS].reshape(k, P.COVER_FEATS)
    pf = obs[k * P.COVER_FEATS: k * P.COVER_FEATS + m * P.PROMPT_FEATS].reshape(m, P.PROMPT_FEATS)
    kk = int(np.argmax(cf @ a[: P.COVER_FEATS]))
    mm = int(np.argmax(pf @ a[P.COVER_FEATS: P.COVER_FEATS + P.PROMPT_FEATS]))
    s = int(np.clip(np.round((a[-1] + 1) / 2 * (len(S.sizes) - 1)), 0, len(S.sizes) - 1))
    return kk, mm, s


# ---- fixed rules -----------------------------------------------------------

def _heuristic_cover_prompt(S, ep):
    k = int(np.argmax(S.base.covers.loc[ep.covers, "log_texture"].values))
    m = int(np.argmax(S.base.prompt_feats[ep.prompts, 0]))
    return k, m


def strat_min_text(S, ep, rng):
    return (*_heuristic_cover_prompt(S, ep), 0, 1)


def strat_max_text(S, ep, rng):
    return (*_heuristic_cover_prompt(S, ep), len(S.sizes) - 1, 1)


def strat_random(S, ep, rng):
    return int(rng.integers(len(ep.covers))), int(rng.integers(len(ep.prompts))), int(rng.integers(len(S.sizes))), 1


def strat_grid(S, ep, rng, w="A"):
    best, arg = 2.0, (0, 0, 0)
    for k in range(len(ep.covers)):
        for m in range(len(ep.prompts)):
            for s in range(len(S.sizes)):
                v, _ = evaluate(S, ep, k, m, s, w)
                if v < best:
                    best, arg = v, (k, m, s)
    return (*arg, len(ep.covers) * len(ep.prompts) * len(S.sizes))


def make_env(S: SplitTables, cover_split: str, prompt_ids: List[int], seed: int, warden: str = "A"):
    import gymnasium as gym
    from gymnasium import spaces

    class SplitEnv(gym.Env):
        def __init__(self):
            self.rng = np.random.default_rng(seed)
            ep = make_episode(S, self.rng, cover_split, prompt_ids)
            self.observation_space = spaces.Box(-10, 10, shape=(len(observation(S, ep)),), dtype=np.float32)
            self.action_space = spaces.Box(-1, 1, shape=(P.COVER_FEATS + P.PROMPT_FEATS + 1,), dtype=np.float32)

        def reset(self, seed=None, options=None):
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            self.ep = make_episode(S, self.rng, cover_split, prompt_ids)
            return observation(S, self.ep), {}

        def step(self, act):
            k, m, s = choice_from_scores(S, self.ep, act)
            v, _ = evaluate(S, self.ep, k, m, s, warden)
            return observation(S, self.ep), -v, True, False, {}

    return SplitEnv()
