"""Pilot experiment: image channel quality, recovery and detectability.

For every cover crop (sources A and B), every method (lsbr, lsbm, hill) and
every rate (bits per colour value), embed a random payload, then record:
PSNR, SSIM, change rate, exact recovery, the SPA estimate and Aletheia's
pretrained EfficientNet scores (the detector trained for that method, plus the
HILL detector applied to everything).

Random payload bytes stand in for whitened fragments, which look random.
Output: results/image_measurements.csv (one row per cover x method x rate,
plus rate-0 rows for the clean covers themselves).

Usage:  python scripts/exp_image.py [--limit N]
"""

import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cmstego import image_channel as ic  # noqa: E402
from cmstego.detectors.image_det import EffNetDetector, spa_score, spam_features  # noqa: E402

RATES = [0.05, 0.1, 0.2, 0.4]
METHODS = ["lsbr", "lsbm", "hill"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--covers", default="data/covers")
    ap.add_argument("--out", default="results/image_measurements.csv")
    ap.add_argument("--limit", type=int, default=None, help="covers per source (for a quick run)")
    args = ap.parse_args()

    paths = []
    for src in sorted(os.listdir(args.covers)):
        ps = sorted(glob.glob(os.path.join(args.covers, src, "*.png")))
        paths += [(src, p) for p in ps[: args.limit]]
    print(f"{len(paths)} covers")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows, images, refused = [], [], []
    rng = np.random.default_rng(2026)
    t0 = time.time()
    for n, (src, p) in enumerate(paths):
        cover = np.asarray(Image.open(p).convert("RGB"))
        key = f"cover-{n}".encode()
        texture = float(np.mean(1.0 / ic.hill_costs(cover[..., 1])))  # higher = busier image
        base = {"source": src, "cover": os.path.basename(p), "texture": texture}
        rows.append({**base, "method": "none", "rate": 0.0, "psnr_db": "", "ssim": "", "change_rate": 0.0, "recovered": ""})
        images.append(cover)
        for m in METHODS:
            for r in RATES:
                nbytes = max(1, int(r * cover.size / 8) - 8)
                data = rng.integers(0, 256, nbytes, dtype=np.uint8).tobytes()
                if 8 * (nbytes + 8) > ic.capacity_bits(cover, m) + 64:
                    # Honest capacity limit: this cover cannot carry this payload.
                    refused.append({**base, "method": m, "rate": r, "status": "over_capacity"})
                    continue
                stego = ic.embed(cover, data, key, m, verify=False)
                ok = ic.extract(stego, key, m) == data  # checked here, recorded, never assumed
                rows.append({**base, "method": m, "rate": r, **ic.quality(cover, stego), "recovered": int(ok)})
                images.append(stego)
        if n % 10 == 0:
            print(f"  embedded {n+1}/{len(paths)} covers ({time.time()-t0:.0f}s)", flush=True)

    # Keep the stego images so detectors can be re-run without re-embedding.
    np.savez_compressed(args.out.replace(".csv", "_images.npz"), images=np.stack(images))
    print("scoring with SPA and extracting SPAM features ...", flush=True)
    feats = []
    for row, im in zip(rows, images):
        row["spa"] = spa_score(im)
        feats.append(spam_features(im))
    np.save(args.out.replace(".csv", "_spam.npy"), np.asarray(feats, dtype=np.float32))

    for det_name in ["lsbr", "lsbm", "hill"]:
        print(f"scoring with EfficientNet-{det_name} ...", flush=True)
        det = EffNetDetector(det_name, size=256)
        scores = det.score(images, batch=32)
        for row, s in zip(rows, scores):
            row[f"effnet_{det_name}"] = float(s)
        del det

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(args.out.replace(".csv", "_refused.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "cover", "texture", "method", "rate", "status"])
        w.writeheader()
        w.writerows(refused)
    print(f"wrote {len(rows)} rows ({len(refused)} over-capacity attempts listed separately) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
