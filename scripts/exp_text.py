"""Proof-of-concept experiment: text channel capacity, KL, recovery and detectability.

For each prompt and each OD budget C, hide random payloads (standing in for
whitened fragments) and decode them again. For every stego-text, also sample a
*cover* text from the same prompt with ordinary top-k sampling, cut to the same
number of tokens, so length cannot give the stego-text away.

Then two text detectors are trained per C on a training half (split by
message, prompts appear in both halves) and evaluated on the held-out half:
  - NgramDetector (black box)
  - LikelihoodDetector (white box: has the same language model)

Outputs: results/text_samples.jsonl, results/text_measurements.csv,
         results/text_detection.csv

Usage:  python scripts/exp_text.py --model data/toy-lm   (or --model gpt2)
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cmstego import text_channel as tc  # noqa: E402
from cmstego.detectors.text_det import LikelihoodDetector, NgramDetector  # noqa: E402
from cmstego.warden import auc  # noqa: E402

from cmstego.prompts import PROMPTS  # noqa: E402

CS = [0.0, 0.02, 0.05, 0.1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/toy-lm")
    ap.add_argument("--per-cell", type=int, default=20, help="messages per (prompt, C)")
    ap.add_argument("--payload", type=int, default=24, help="bytes per message")
    ap.add_argument("--out", default="results")
    ap.add_argument("--variant", default="retry", choices=["clear", "filter", "retry", "nlfix"],
                    help="clear: subtext-codec framing as-is (no nonce, filter guard); "
                         "filter: nonce+whitening, filter guard; retry: nonce+whitening, retry guard; "
                         "nlfix: filter + line-break fix (blank-line tokens banned, line breaks shown doubled)")
    ap.add_argument("--cs", default="0,0.02,0.05,0.1")
    args = ap.parse_args()
    CS = [float(c) for c in args.cs.split(",")]
    key = None if args.variant == "clear" else b"pilot-shared-key"
    guard = "retry" if args.variant == "retry" else "filter"
    tag = f"text_{args.variant}_"
    model, tok = tc.load_lm(args.model)
    rng = random.Random(7)

    samples, meas, failures = [], [], []
    t0 = time.time()
    for C in CS:
        for pi, prompt in enumerate(PROMPTS):
            cfg = tc.TextConfig(args.model, prompt, C=C, max_new_tokens=3000, guard=guard,
                                newline_fix=args.variant == "nlfix")
            for j in range(args.per_cell):
                data = bytes(rng.randrange(256) for _ in range(args.payload))
                t1 = time.time()
                try:
                    text, st = tc.encode(data, cfg, model, tok, key=key)
                except ValueError as exc:  # e.g. retry guard found no clean re-tokenisation
                    failures.append({"C": C, "prompt_id": pi, "j": j, "reason": str(exc)[:120]})
                    continue
                enc_s = time.time() - t1
                t1 = time.time()
                ok = tc.decode(text, cfg, model, tok, key=key) == data
                dec_s = time.time() - t1
                cover = tc.sample_cover(cfg, model, tok, st.tokens, seed=hash((C, pi, j)) & 0xFFFFFFFF)
                samples.append({"C": C, "prompt": prompt, "j": j, "stego": text, "cover": cover})
                meas.append({"C": C, "prompt_id": pi, "j": j, "payload_bytes": args.payload, "tokens": st.tokens,
                             "bits_per_token": st.bits_per_token, "kl_bits_per_token": st.kl_bits_per_token,
                             "kl_bits_total": st.kl_bits_total, "kl_vs_topk_bits_per_token": st.kl_vs_topk_bits_per_token,
                             "attempts": st.attempts, "mean_entropy_bits": st.mean_entropy_bits,
                             "mean_temperature": st.mean_temperature, "recovered": int(ok),
                             "encode_s": enc_s, "decode_s": dec_s})
        print(f"C={C}: done ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, tag + "samples.jsonl"), "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    with open(os.path.join(args.out, tag + "measurements.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(meas[0].keys()))
        w.writeheader()
        w.writerows(meas)

    with open(os.path.join(args.out, tag + "encode_failures.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["C", "prompt_id", "j", "reason"])
        w.writeheader()
        w.writerows(failures)
    print(f"encode failures: {len(failures)}", flush=True)

    # ---- detection: train on even j, test on odd j ----
    prompt_of = {}
    for s in samples:
        prompt_of[s["stego"]] = s["prompt"]
        prompt_of[s["cover"]] = s["prompt"]
    det_rows, score_rows = [], []
    for C in CS:
        cell = [s for s in samples if s["C"] == C]
        tr = [s for s in cell if s["j"] % 2 == 0]
        te = [s for s in cell if s["j"] % 2 == 1]
        Xtr = [s["cover"] for s in tr] + [s["stego"] for s in tr]
        ytr = [0] * len(tr) + [1] * len(tr)
        for name, det in [("ngram", NgramDetector()),
                          ("likelihood", LikelihoodDetector(lambda t: prompt_of[t], model, tok))]:
            det.fit(Xtr, ytr)
            sc = det.score([s["cover"] for s in te])
            ss = det.score([s["stego"] for s in te])
            det_rows.append({"C": C, "detector": name, "n_test_pairs": len(te), "auc": auc(sc, ss)})
            for s, a, b in zip(te, sc, ss):
                score_rows.append({"C": C, "detector": name, "prompt": s["prompt"], "j": s["j"],
                                   "cover_score": float(a), "stego_score": float(b)})
            print(f"C={C} {name}: AUC {det_rows[-1]['auc']:.3f} on {len(te)} held-out pairs", flush=True)
    with open(os.path.join(args.out, tag + "detection.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(det_rows[0].keys()))
        w.writeheader()
        w.writerows(det_rows)
    with open(os.path.join(args.out, tag + "heldout_scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(score_rows[0].keys()))
        w.writeheader()
        w.writerows(score_rows)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
