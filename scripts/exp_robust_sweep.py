"""Robust method settings (layout, grid spacing): survival vs pixels changed vs SPAM detection.
Usage: python scripts/exp_robust_sweep.py [image_bytes]   (default 200; writes results/robust_sweep_<n>B.csv)"""
import glob, io, os, sys, json
import numpy as np
from PIL import Image
sys.path.insert(0, "src")
from cmstego import image_channel as ic
from cmstego.detectors.image_det import spam_features, spa_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "scripts")
import exp_robustness as R

paths = sorted(glob.glob("data/covers/A_kodak/*.png"))
covers = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
groups = [os.path.basename(p).split("_")[0] for p in paths]
clean_feats = np.array([spam_features(c) for c in covers])
ATT = ["none", "jpeg-90", "jpeg-75", "gauss-1", "sp-1%", "median-3"]
size = int(sys.argv[1]) if len(sys.argv) > 1 else 200
configs = [(False, 28.0), (True, 28.0), (True, 20.0), (True, 14.0)]
out = []
for packed, delta in configs:
    ic.ROBUST_PACKED, ic.ROBUST_DELTA = packed, delta
    rng = np.random.default_rng(1)
    stegos, rec = [], {a: [] for a in ATT}
    changed, psnr = [], []
    for c in covers:
        d = rng.bytes(size)
        s = ic.embed(c, d, b"sweep", "robust")
        stegos.append(s)
        q = ic.quality(c, s); changed.append(q["change_rate"]); psnr.append(q["psnr_db"])
    for i, s in enumerate(stegos[:48]):  # survival on 48 crops
        rng2 = np.random.default_rng(i)
        for a in ATT:
            try:
                ok = ic.extract(R.attack(s, a, rng2), b"sweep", "robust") is not None
            except Exception:
                ok = False
            rec[a].append(ok)
    feats = np.array([spam_features(s) for s in stegos])
    X = np.r_[clean_feats, feats]; y = np.r_[np.zeros(len(covers)), np.ones(len(covers))]; g = groups + groups
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1))
    pr = cross_val_predict(clf, X, y, groups=g, cv=GroupKFold(5), method="predict_proba")[:, 1]
    auc = roc_auc_score(y, pr)
    neg = pr[: len(covers)]; pos = pr[len(covers):]
    caught = float((pos > np.quantile(neg, 0.95, method="higher")).mean())
    row = {"layout": "packed" if packed else "spread", "delta": delta, "psnr_db": float(np.mean(psnr)),
           "changed_pct": 100 * float(np.mean(changed)), "spam_auc": float(auc), "caught_5fa": caught,
           **{f"surv_{a}": 100 * float(np.mean(v)) for a, v in rec.items()}}
    out.append(row); print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
import csv
os.makedirs("results", exist_ok=True)
with open(f"results/robust_sweep_{size}B.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
