"""Summarise a text-experiment folder: capacity, KL, and detector AUCs with 95% bootstrap CIs.

    python scripts/summarize_text.py results_gpt2
"""

import os
import sys

import numpy as np
import pandas as pd


def auc(neg, pos):
    neg, pos = np.asarray(neg), np.asarray(pos)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return gt + 0.5 * eq


def boot_ci(neg, pos, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(neg)
    vals = [auc(neg[i], pos[i]) for i in (rng.integers(0, n, n) for _ in range(reps))]
    return np.percentile(vals, [2.5, 97.5])


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results_gpt2"
    for variant in ("filter", "clear", "retry"):
        mpath = os.path.join(d, f"text_{variant}_measurements.csv")
        spath = os.path.join(d, f"text_{variant}_heldout_scores.csv")
        if not os.path.exists(mpath):
            continue
        m = pd.read_csv(mpath)
        print(f"\n=== {variant} ({len(m)} messages) ===")
        cols = [c for c in ["bits_per_token", "kl_bits_per_token", "kl_vs_topk_bits_per_token"] if c in m]
        print(m.groupby("C")[cols].mean().round(3).to_string())
        if os.path.exists(spath):
            s = pd.read_csv(spath)
            for (C, det), g in s.groupby(["C", "detector"]):
                neg, pos = g["cover_score"].to_numpy(), g["stego_score"].to_numpy()
                lo, hi = boot_ci(neg, pos)
                print(f"  C={C:<4} {det:<10} AUC {auc(neg, pos):.2f} [{lo:.2f}, {hi:.2f}]  ({len(g)} pairs)")


if __name__ == "__main__":
    main()
