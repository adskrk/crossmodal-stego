"""Receiver only: recover the secret from a saved stego-text and stego-image.

    python scripts/receive.py --model gpt2
    python scripts/receive.py --model gpt2 --key wrong-key     # shows a wrong key fails

Reads demo_out/stego.txt and demo_out/stego.png by default. Key, prompt and
method must match what the sender used (the demo's defaults).
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cmstego import system, text_channel  # noqa: E402
from cmstego.prompts import PROMPTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="demo_out/stego.txt")
    ap.add_argument("--image", default="demo_out/stego.png")
    ap.add_argument("--model", default=os.environ.get("CMSTEGO_LM", "data/toy-lm"))
    ap.add_argument("--prompt", default=PROMPTS[1])
    ap.add_argument("--method", default="hill", choices=["lsbr", "lsbm", "hill"])
    ap.add_argument("--C", type=float, default=0.0)
    ap.add_argument("--key", default="demo-shared-key")
    args = ap.parse_args()

    print(f"Loading language model: {args.model}")
    model, tok = text_channel.load_lm(args.model)
    with open(args.text, encoding="utf-8") as f:
        text = f.read()
    image = np.asarray(Image.open(args.image).convert("RGB"))
    cfg = system.SendConfig(key=args.key.encode(), model_path=args.model, prompt=args.prompt,
                            image_method=args.method, C=args.C)

    t = time.time()
    rep = system.receive(text, image, cfg, model, tok)
    print(f"\nRECEIVER ({time.time()-t:.1f} s)  text fragment ok: {rep.text_ok}, image fragment ok: {rep.image_ok}")
    if rep.ok:
        print(f"  recovered secret: {rep.message.decode(errors='replace')!r}")
    else:
        print(f"  FAILED: {rep.reason}")


if __name__ == "__main__":
    main()
