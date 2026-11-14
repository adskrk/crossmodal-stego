"""Diagnose what the white-box text detector keys on at C = 0.

Computes each likelihood feature's single-feature ROC-AUC (stego vs cover) and
how many texts contain a token that, when the text is re-read by the model's
tokenizer, falls outside the model's top-k. Ordinary samples sometimes do
(sampled tokens can merge into a different tokenisation); stego-text never does,
because exact decoding needs a clean re-tokenisation.

Writes results/text_fingerprint.csv.
Usage:  python scripts/diag_text_fingerprint.py --model data/toy-lm --variant filter
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cmstego import text_channel as tc  # noqa: E402
from cmstego.detectors.text_det import likelihood_features  # noqa: E402

NAMES = ["mean log-prob", "std log-prob", "10th pct log-prob", "median log-prob",
         "share of tokens outside top-k", "share of top-1 tokens", "mean surprise minus entropy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/toy-lm")
    ap.add_argument("--variant", default="filter")
    ap.add_argument("--C", type=float, default=0.0)
    args = ap.parse_args()
    model, tok = tc.load_lm(args.model)
    S = [json.loads(l) for l in open(f"results/text_{args.variant}_samples.jsonl", encoding="utf-8")]
    S = [s for s in S if s["C"] == args.C]
    Fc = np.array([likelihood_features([s["cover"]], s["prompt"], model, tok)[0] for s in S])
    Fs = np.array([likelihood_features([s["stego"]], s["prompt"], model, tok)[0] for s in S])
    y = np.r_[np.zeros(len(S)), np.ones(len(S))]
    rows = []
    for i, n in enumerate(NAMES):
        a = roc_auc_score(y, np.r_[Fc[:, i], Fs[:, i]])
        rows.append({"feature": n, "cover_mean": Fc[:, i].mean(), "stego_mean": Fs[:, i].mean(),
                     "single_feature_auc": max(a, 1 - a)})
    rows.append({"feature": "texts with >=1 token outside top-k", "cover_mean": int((Fc[:, 4] > 0).sum()),
                 "stego_mean": int((Fs[:, 4] > 0).sum()), "single_feature_auc": len(S)})
    with open("results/text_fingerprint.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
