"""What happens to the hidden image data when the stego image is changed in transit?

For each image method and payload size, embed random payloads into cover crops,
apply one attack, and record:
  * recovered  — the receiver got the exact payload back (CRC and, for the robust
                 method, error correction included);
  * bit errors — share of the embedded bits that came out wrong, read with the
                 known length (0.5 = random, the data is gone).

Attacks (each a common real-world cause of change):
  jpeg-95/90/75      re-saving as JPEG (chat apps, social media)
  gauss-0.5/1/2      Gaussian noise with that standard deviation (sensor grain)
  sp-0.1%/1%         salt & pepper: that share of colour values set to 0 or 255
  median-3           3x3 median filter (noise reduction)

Usage:  python scripts/exp_robustness.py [--methods lsbr,lsbm,hill,robust] [--covers 24]
Writes results/robustness.csv and prints a summary.
"""

import argparse
import glob
import io
import os
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cmstego import image_channel as ic  # noqa: E402

KEY = b"robustness-key"


def attack(img: np.ndarray, name: str, rng: np.random.Generator) -> np.ndarray:
    if name == "none":
        return img
    if name.startswith("jpeg-"):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=int(name.split("-")[1]))
        return np.asarray(Image.open(buf).convert("RGB"))
    if name.startswith("gauss-"):
        s = float(name.split("-")[1])
        return np.clip(np.round(img + rng.normal(0, s, img.shape)), 0, 255).astype(np.uint8)
    if name.startswith("sp-"):
        p = float(name.split("-")[1].rstrip("%")) / 100
        out = img.copy()
        hit = rng.random(img.shape) < p
        out[hit] = rng.choice([0, 255], size=int(hit.sum())).astype(np.uint8)
        return out
    if name == "median-3":
        return np.stack([ndimage.median_filter(img[..., k], size=3) for k in range(3)], axis=-1)
    raise ValueError(name)


ATTACKS = ["none", "jpeg-95", "jpeg-90", "jpeg-75", "gauss-0.5", "gauss-1", "gauss-2", "sp-0.1%", "sp-1%", "median-3"]


def raw_bits(stego: np.ndarray, method: str, nbits: int) -> np.ndarray:
    """The embedded bit positions read back with the known length (no checks)."""
    if method in ("lsbr", "lsbm"):
        flat = stego.reshape(-1)
        order = np.random.default_rng(ic._seed(KEY, "pos")).permutation(flat.size)[:nbits]
        return (flat[order] & 1).astype(np.uint8)
    if method == "hill":
        from pystc import pystc as core
        x = stego.reshape(-1).astype(np.int32)
        xs = x[ic._stc_order(x.size, KEY)]
        return np.unpackbits(np.frombuffer(bytes(core._unhide_stc(xs[ic.STC_HEADER_REGION:], nbits)), np.uint8))
    if method == "robust":
        return ic.robust_raw_bits(stego, KEY, nbits)
    raise ValueError(method)


def sent_bits(data: bytes, method: str) -> np.ndarray:
    if method == "robust":
        return ic.robust_coded_bits(data)
    return np.unpackbits(np.frombuffer(ic._frame(data), np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="lsbr,lsbm,hill")
    ap.add_argument("--covers", type=int, default=24)
    ap.add_argument("--sizes", default="64,400", help="payload bytes")
    ap.add_argument("--covers-dir", default="data/covers/A_kodak")
    ap.add_argument("--out", default="results/robustness.csv")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.covers_dir, "*.png")))
    paths = paths[:: max(1, len(paths) // args.covers)][: args.covers]
    rng = np.random.default_rng(0)
    rows = []
    t0 = time.time()
    for method in args.methods.split(","):
        for size in (int(s) for s in args.sizes.split(",")):
            for p in paths:
                cover = np.asarray(Image.open(p).convert("RGB"))
                data = rng.bytes(size)
                try:
                    stego = ic.embed(cover, data, KEY, method)
                except ValueError:
                    rows.append({"method": method, "bytes": size, "cover": os.path.basename(p), "attack": "embed",
                                 "recovered": 0, "bit_errors": np.nan, "psnr_db": np.nan})
                    continue
                want = sent_bits(data, method)
                psnr = ic.quality(cover, stego)["psnr_db"]
                for a in ATTACKS:
                    got_img = attack(stego, a, rng)
                    try:
                        ok = int(ic.extract(got_img, KEY, method) == data)
                    except Exception:
                        ok = 0
                    try:
                        got = raw_bits(got_img, method, len(want))
                        ber = float((got[: len(want)] != want).mean())
                    except Exception:
                        ber = 0.5
                    rows.append({"method": method, "bytes": size, "cover": os.path.basename(p), "attack": a,
                                 "recovered": ok, "bit_errors": ber, "psnr_db": psnr})
            print(f"{method} {size} B done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    ok = df[df.attack != "embed"]
    tab = ok.pivot_table(index=["method", "bytes"], columns="attack", values="recovered", aggfunc="mean")[ATTACKS]
    print("\nShare of payloads recovered exactly:")
    print((100 * tab).round(0).astype(int).to_string())
    tab = ok.pivot_table(index=["method", "bytes"], columns="attack", values="bit_errors", aggfunc="mean")[ATTACKS]
    print("\nMean share of wrong bits (0.5 = data gone):")
    print(tab.round(3).to_string())
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
