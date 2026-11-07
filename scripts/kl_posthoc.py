"""Recompute KL diagnostics (vs full model and vs ordinary top-k sampling) from the
stego-texts saved by exp_text.py, by re-decoding them. Writes results/text_kl.csv.

Usage:  python scripts/kl_posthoc.py --model data/toy-lm [--per-c 48]
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cmstego import text_channel as tc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/toy-lm")
    ap.add_argument("--per-c", type=int, default=48)
    args = ap.parse_args()
    model, tok = tc.load_lm(args.model)
    samples = [json.loads(l) for l in open("results/text_samples.jsonl")]
    taken = defaultdict(int)
    rows = []
    for s in samples:
        if taken[s["C"]] >= args.per_c:
            continue
        taken[s["C"]] += 1
        cfg = tc.TextConfig(args.model, s["prompt"], C=s["C"], max_new_tokens=None)
        _, st = tc.decode_with_stats(s["stego"], cfg, model, tok)
        rows.append({"C": s["C"], "prompt": s["prompt"], "j": s["j"], **st})
    with open("results/text_kl.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to results/text_kl.csv")


if __name__ == "__main__":
    main()
