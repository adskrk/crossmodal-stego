"""Live demo: hide one secret across a stego-text and a stego-image, then recover it.

  python scripts/demo.py --secret "Meet at the library at 5pm. Bring the notes." \
                         --cover data/covers/A_kodak/05_02.png --model gpt2

Writes demo_out/stego.txt and demo_out/stego.png, prints what each channel
carried, image quality, timings, and the receiver's verdict. With --attack it
also shows what happens when a carrier is damaged:
  --attack jpeg   re-save the image as JPEG quality 90 (image fragment lost)
  --attack edit   change one word in the text (text fragment lost)
"""

import argparse
import io
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cmstego import system, text_channel  # noqa: E402
from cmstego.detectors.image_det import spa_score  # noqa: E402
from cmstego.prompts import PROMPTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secret", default="Meet at the library at 5pm. Bring the notes for the BTP review, "
                                        "and do not tell anyone else about this plan.")
    ap.add_argument("--cover", default="data/covers/A_kodak/05_02.png")
    ap.add_argument("--model", default=os.environ.get("CMSTEGO_LM", "data/toy-lm"))
    ap.add_argument("--prompt", default=PROMPTS[1])
    ap.add_argument("--method", default="hill", choices=["lsbr", "lsbm", "hill"])
    ap.add_argument("--C", type=float, default=0.0)
    ap.add_argument("--key", default="demo-shared-key")
    ap.add_argument("--max-text-tokens", type=int, default=150)
    ap.add_argument("--attack", choices=["none", "jpeg", "edit"], default="none")
    ap.add_argument("--out", default="demo_out")
    args = ap.parse_args()

    print(f"Loading language model: {args.model}")
    model, tok = text_channel.load_lm(args.model)
    cover = np.asarray(Image.open(args.cover).convert("RGB"))
    cfg = system.SendConfig(key=args.key.encode(), model_path=args.model, prompt=args.prompt,
                            image_method=args.method, C=args.C, max_text_tokens=args.max_text_tokens)
    secret = args.secret.encode()

    print(f"\nSENDER  secret: {len(secret)} bytes")
    car = system.send(secret, cover, cfg, model, tok)
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, "stego.txt"), "w").write(car.text)
    Image.fromarray(car.image).save(os.path.join(args.out, "stego.png"))
    print(f"  text  channel: {car.text_bytes} secret bytes in {car.text_stats.tokens} tokens "
          f"({car.text_stats.bits_per_token:.2f} payload bits/token incl. header; "
          f"KL {car.text_stats.kl_bits_per_token:.4f} bits/token)")
    print(f"  image channel: {car.image_bytes} secret bytes, method {args.method}, "
          f"PSNR {car.image_quality['psnr_db']:.1f} dB, SSIM {car.image_quality['ssim']:.4f}, "
          f"{100*car.image_quality['change_rate']:.2f}% of colour values changed")
    print(f"  SPA estimate of LSB-replacement payload: cover {spa_score(cover):.3f}, stego {spa_score(car.image):.3f}")
    print(f"  sender time: {car.seconds:.1f} s")
    print("\n--- stego-text -------------------------------------------")
    print(car.text)
    print("----------------------------------------------------------")

    text, image = car.text, car.image
    if args.attack == "jpeg":
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="JPEG", quality=90)
        image = np.asarray(Image.open(buf).convert("RGB"))
        print("\nATTACK  image re-saved as JPEG (quality 90)")
    elif args.attack == "edit":
        words = text.split(" ")
        i = len(words) // 2
        words[i] = "banana"
        text = " ".join(words)
        print("\nATTACK  one word of the text replaced")

    t = time.time()
    rep = system.receive(text, image, cfg, model, tok)
    print(f"\nRECEIVER ({time.time()-t:.1f} s)  text fragment ok: {rep.text_ok}, image fragment ok: {rep.image_ok}")
    if rep.ok:
        print(f"  recovered: {rep.message.decode()!r}")
        print(f"  exact match: {rep.message == secret}")
    else:
        print(f"  FAILED: {rep.reason}")


if __name__ == "__main__":
    main()
