"""The price of robustness: how detectable is the robust method compared with HILL?

Embeds the same payload size with HILL + STC and with the robust prototype into
the Kodak crops, then scores clean and stego images with
  * SPA (no training), and
  * SPAM features + logistic regression, trained and scored out-of-fold by photo
    (no photo is scored by a model that saw it),
and reports ROC-AUC with 95% bootstrap intervals, P_E, detection at 5% false
alarms, PSNR and the share of colour values changed.

Usage:  python scripts/exp_robust_detect.py [--bytes 200]
Writes results/robust_detect.csv and results/robust_detect_summary.csv.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cmstego import image_channel as ic  # noqa: E402
from cmstego.detectors.image_det import spa_score, spam_features  # noqa: E402


def auc_ci(neg, pos, n=1000, seed=0):
    neg, pos = np.asarray(neg), np.asarray(pos)
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    a = roc_auc_score(y, np.r_[neg, pos])
    rng = np.random.default_rng(seed)
    bs = [roc_auc_score(y, np.r_[rng.choice(neg, len(neg)), rng.choice(pos, len(pos))]) for _ in range(n)]
    return a, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def operating_points(neg, pos):
    neg, pos = np.sort(np.asarray(neg, float)), np.asarray(pos, float)
    thr = np.unique(np.r_[neg, pos, np.inf])
    pe = float(np.min([((neg >= t).mean() + (pos < t).mean()) / 2 for t in thr]))
    return pe, float((pos > np.quantile(neg, 0.95, method="higher")).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=200)
    ap.add_argument("--covers-dir", default="data/covers/A_kodak")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    rows = []
    for p in sorted(glob.glob(os.path.join(args.covers_dir, "*.png"))):
        cover = np.asarray(Image.open(p).convert("RGB"))
        group = os.path.basename(p).split("_")[0]
        rows.append({"cover": os.path.basename(p), "group": group, "method": "none", "img": cover})
        data = rng.bytes(args.bytes)
        for m in ("hill", "robust"):
            try:
                st = ic.embed(cover, data, b"detect-key", m)
            except ValueError:
                continue
            q = ic.quality(cover, st)
            rows.append({"cover": os.path.basename(p), "group": group, "method": m, "img": st,
                         "psnr_db": q["psnr_db"], "change_rate": q["change_rate"]})
    df = pd.DataFrame(rows)
    df["spa"] = [spa_score(x) for x in df.img]
    feats = np.array([spam_features(x) for x in df.img])
    out = []
    for m in ("hill", "robust"):
        keep = df.method.isin(["none", m]).values
        sub = df[keep].reset_index(drop=True)
        y = (sub.method == m).astype(int).values
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1))
        sub["spam"] = cross_val_predict(clf, feats[keep], y, groups=sub.group, cv=GroupKFold(5), method="predict_proba")[:, 1]
        for det in ("spa", "spam"):
            neg, pos = sub[sub.method == "none"][det], sub[sub.method == m][det]
            a, lo, hi = auc_ci(neg, pos)
            pe, tpr5 = operating_points(neg, pos)
            out.append({"method": m, "detector": det, "auc": a, "lo": lo, "hi": hi, "pe": pe, "caught_at_5pct_fa": tpr5,
                        "n_stego": len(pos), "psnr_db": sub[sub.method == m].psnr_db.mean(),
                        "change_rate": sub[sub.method == m].change_rate.mean()})
    res = pd.DataFrame(out)
    os.makedirs(args.out, exist_ok=True)
    res.to_csv(os.path.join(args.out, "robust_detect_summary.csv"), index=False)
    df.drop(columns=["img"]).to_csv(os.path.join(args.out, "robust_detect.csv"), index=False)
    print(f"payload {args.bytes} bytes per 256x256 crop")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
