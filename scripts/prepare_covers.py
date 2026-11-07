"""Cut cover images into 256x256 RGB PNG crops.

Two cover *sources* are kept separate on purpose, because steganalysis
detectors are known to lose accuracy when tested on a different image source
than they were trained on (cover-source mismatch):

* source A: Kodak Lossless True Color Image Suite (24 photos, 768x512)
* source B: colour photos bundled with scikit-image (astronaut, coffee, ...)

On your own machine you can add more sources (ALASKA2, DIV2K) with --extra.

Usage:  python scripts/prepare_covers.py --kodak <dir with kodak PNGs> --out data/covers
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image

SKIMAGE_COLOUR = ["astronaut.png", "chelsea.png", "coffee.png", "rocket.jpg", "motorcycle_left.png",
                  "motorcycle_right.png", "retina.jpg", "ihc.png", "hubble_deep_field.jpg"]


def crops(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    for y in range(0, h - size + 1, size):
        for x in range(0, w - size + 1, size):
            yield img[y : y + size, x : x + size]


def write_source(paths, out_dir, size):
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for p in sorted(paths):
        img = np.asarray(Image.open(p).convert("RGB"))
        stem = os.path.splitext(os.path.basename(p))[0]
        for i, c in enumerate(crops(img, size)):
            Image.fromarray(c).save(os.path.join(out_dir, f"{stem}_{i:02d}.png"))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kodak", required=True)
    ap.add_argument("--out", default="data/covers")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--extra", default=None, help="optional folder of more images (source C)")
    args = ap.parse_args()

    na = write_source(glob.glob(os.path.join(args.kodak, "*.png")), os.path.join(args.out, "A_kodak"), args.size)
    import skimage

    sk = os.path.join(os.path.dirname(skimage.__file__), "data")
    nb = write_source([os.path.join(sk, f) for f in SKIMAGE_COLOUR if os.path.exists(os.path.join(sk, f))],
                      os.path.join(args.out, "B_skimage"), args.size)
    print(f"source A (Kodak): {na} crops; source B (scikit-image): {nb} crops")
    if args.extra:
        nc = write_source(glob.glob(os.path.join(args.extra, "*")), os.path.join(args.out, "C_extra"), args.size)
        print(f"source C (extra): {nc} crops")


if __name__ == "__main__":
    main()
