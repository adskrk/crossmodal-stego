"""Time the sender: original bisection ("exact") vs capacity-sized search ("fast").

    python scripts/bench_send.py --model gpt2

Runs each secret size with both modes on the same cover and prompt, checks the
receiver recovers the secret exactly, and prints a table. Takes a few minutes
with GPT-2 on a laptop CPU.
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


def load_cover(path):
    if path and os.path.exists(path):
        im = Image.open(path).convert("RGB")
    else:
        from skimage import data
        im = Image.fromarray(data.astronaut())
    im.thumbnail((512, 512))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("CMSTEGO_LM", "data/toy-lm"))
    ap.add_argument("--cover", default="data/covers/A_kodak/05_02.png")
    ap.add_argument("--sizes", default="27,1000", help="secret sizes in bytes, comma separated")
    ap.add_argument("--max-text-tokens", type=int, default=100)
    args = ap.parse_args()

    model, tok = text_channel.load_lm(args.model)
    cover = load_cover(args.cover)
    calls = {"n": 0}
    orig = text_channel._encode_once

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    text_channel._encode_once = counting
    base = b"Meet at the library at 5pm. Bring the notes for the review. "
    print(f"model {args.model}, cover {cover.shape[1]}x{cover.shape[0]}, max {args.max_text_tokens} text tokens\n")
    print(f"{'secret':>7} {'mode':>6} {'send s':>7} {'encodes':>8} {'text B':>7} {'image B':>8} {'recv s':>7}  exact")
    for n in [int(x) for x in args.sizes.split(",")]:
        secret = (base * (n // len(base) + 1))[:n]
        for mode in ("exact", "fast"):
            cfg = system.SendConfig(key=b"bench-key", model_path=args.model, prompt=PROMPTS[1],
                                    max_text_tokens=args.max_text_tokens, fill=mode)
            calls["n"] = 0
            t = time.time()
            car = system.send(secret, cover, cfg, model, tok, msg_id=1)
            ts = time.time() - t
            enc = calls["n"]
            t = time.time()
            rep = system.receive(car.text, car.image, cfg, model, tok)
            tr = time.time() - t
            print(f"{n:7d} {mode:>6} {ts:7.1f} {enc:8d} {car.text_bytes:7d} {car.image_bytes:8d} {tr:7.1f}  "
                  f"{rep.ok and rep.message == secret}", flush=True)


if __name__ == "__main__":
    main()
