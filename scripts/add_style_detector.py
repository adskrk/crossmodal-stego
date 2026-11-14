"""Score saved stego/cover texts with the style detector (no language model needed).

    python scripts/add_style_detector.py --results results_gpt2 --variant filter

Same protocol as exp_text.py: for each budget C, train on even sample indices j
and score the odd ones. Writes text_<variant>_heldout_scores_style.csv next to
the existing held-out scores, so selection experiments can use it as a third
text detector.
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cmstego.detectors.text_det import StyleDetector  # noqa: E402
from cmstego.warden import auc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_gpt2")
    ap.add_argument("--variant", default="filter")
    args = ap.parse_args()
    path = os.path.join(args.results, f"text_{args.variant}_samples.jsonl")
    with open(path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    rows = []
    for C in sorted({s["C"] for s in samples}):
        cell = [s for s in samples if s["C"] == C]
        tr = [s for s in cell if s["j"] % 2 == 0]
        te = [s for s in cell if s["j"] % 2 == 1]
        det = StyleDetector().fit([s["cover"] for s in tr] + [s["stego"] for s in tr], [0] * len(tr) + [1] * len(tr))
        sc, ss = det.score([s["cover"] for s in te]), det.score([s["stego"] for s in te])
        for s, a, b in zip(te, sc, ss):
            rows.append({"C": C, "detector": "style", "prompt": s["prompt"], "j": s["j"],
                         "cover_score": float(a), "stego_score": float(b)})
        print(f"C={C} style: AUC {auc(sc, ss):.3f} on {len(te)} held-out pairs")
    out = os.path.join(args.results, f"text_{args.variant}_heldout_scores_style.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
