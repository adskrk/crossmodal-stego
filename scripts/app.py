"""Browser app for the demo: hide a secret in text + image, then extract it.

    python scripts/app.py --model gpt2

then open http://127.0.0.1:7860 in your browser. The language model is loaded
once at start-up. Same system, same defaults as scripts/demo.py and
scripts/receive.py; this file only adds a user interface.
"""

import argparse
import io
import os
import sys
import time

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cmstego import aont, image_channel, system, text_channel  # noqa: E402
from cmstego.prompts import PROMPTS  # noqa: E402

OUT = "app_out"
SIZES = ["512", "768", "1024", "1536", "original"]  # longest side of the cover, in pixels
METHOD_NAMES = {aont.RAW: "not compressed", aont.DEFLATE: "deflate", aont.BZ2: "bz2"}


def build(model_path):
    print(f"Loading language model: {model_path}")
    model, tok = text_channel.load_lm(model_path)

    def cfg_for(key, method, max_tokens=150):
        return system.SendConfig(key=key.encode(), model_path=model_path, prompt=PROMPTS[1],
                                 image_method=method, C=0.0, max_text_tokens=int(max_tokens))

    def hide(photo, secret, key, method, max_tokens, size):
        if photo is None:
            raise gr.Error("Upload a cover photo first.")
        if not secret:
            raise gr.Error("Type a secret message.")
        im = Image.fromarray(photo).convert("RGB")
        if size != "original":
            im.thumbnail((int(size), int(size)))
        cover = np.asarray(im)
        try:
            car = system.send(secret.encode(), cover, cfg_for(key, method, max_tokens), model, tok)
        except ValueError as e:
            raise gr.Error(f"Could not hide the message: {e}. Try a larger photo size, a more detailed photo, "
                           "or a shorter secret.")
        comp, stored = aont.compress(secret.encode())
        cap_bytes = image_channel.capacity_bits(cover, method) // 8
        os.makedirs(OUT, exist_ok=True)
        png, txt = os.path.join(OUT, "stego.png"), os.path.join(OUT, "stego.txt")
        Image.fromarray(car.image).save(png)
        Image.fromarray(cover).save(os.path.join(OUT, "cover.png"))
        with open(txt, "w", encoding="utf-8") as f:
            f.write(car.text)
        q, s = car.image_quality, car.text_stats
        tail = f" (+{s.tail_tokens} ordinary tokens to finish the sentence)" if s.tail_tokens else ""
        used = f", {100 * car.image_bytes / cap_bytes:.1f}% of this photo's capacity" if cap_bytes else ""
        report = (
            f"Secret: {len(secret.encode())} bytes, {METHOD_NAMES[comp]} -> {len(stored)} bytes, sealed into a "
            f"{car.payload_bytes}-byte all-or-nothing package (neither part opens alone)\n"
            f"Split: {car.text_bytes} bytes in the text + {car.image_bytes} bytes in the image\n"
            f"Text:  {s.tokens} tokens{tail}, {s.bits_per_token:.2f} payload bits/token, KL {s.kl_bits_per_token:.4f} bits/token\n"
            f"Image: {cover.shape[1]}x{cover.shape[0]}, method {method}, PSNR {q['psnr_db']:.1f} dB, "
            f"SSIM {q['ssim']:.4f}, {100*q['change_rate']:.2f}% of colour values changed{used}\n"
            f"Time:  {car.seconds:.1f} s"
        )
        return cover, car.image, car.text, report, [png, txt], png, txt

    def extract(png_file, txt_file, pasted, key, method, jpeg, edit):
        if png_file is None:
            raise gr.Error("Upload the stego image (stego.png).")
        image = np.asarray(Image.open(png_file).convert("RGB"))
        if txt_file is not None:
            with open(txt_file, encoding="utf-8") as f:
                text = f.read()
        elif pasted:
            text = pasted
        else:
            raise gr.Error("Upload stego.txt or paste the stego text.")
        notes = []
        if jpeg:
            buf = io.BytesIO()
            Image.fromarray(image).save(buf, format="JPEG", quality=90)
            image = np.asarray(Image.open(buf).convert("RGB"))
            notes.append("ATTACK: image re-saved as JPEG (quality 90)")
        if edit:
            words = text.split(" ")
            words[len(words) // 2] = "banana"
            text = " ".join(words)
            notes.append("ATTACK: one word of the text replaced")
        t = time.time()
        rep = system.receive(text, image, cfg_for(key, method), model, tok)
        lines = notes + [f"Text fragment ok: {rep.text_ok}    Image fragment ok: {rep.image_ok}    "
                         f"({time.time()-t:.1f} s)"]
        if rep.ok:
            return rep.message.decode(errors="replace"), "\n".join(lines)
        return "", "\n".join(lines + [f"FAILED: {rep.reason}"])

    with gr.Blocks(title="Cross-modal steganography") as app:
        gr.Markdown("## Cross-modal steganography\nOne secret is split between LLM-generated text "
                    f"({model_path}) and an image, and recovered exactly with the shared key.")
        last_png, last_txt = gr.State(), gr.State()
        with gr.Tab("1. Hide"):
            with gr.Row():
                photo = gr.Image(label="Cover photo", type="numpy")
                with gr.Column():
                    secret = gr.Textbox(label="Secret message", lines=3)
                    key = gr.Textbox(label="Shared key", value="demo-shared-key")
                    method = gr.Radio(["hill", "lsbm", "lsbr"], value="hill", label="Image method")
                    max_tokens = gr.Slider(60, 250, value=150, step=10, label="Target text length (tokens)")
                    size = gr.Radio(SIZES, value="512", label="Photo size (longest side, px). Larger holds more, "
                                    "but hides more slowly")
                    hide_btn = gr.Button("Hide secret", variant="primary")
            report = gr.Textbox(label="Measurements", lines=5)
            with gr.Row():
                cover_out = gr.Image(label="Cover", format="png")
                stego_out = gr.Image(label="Stego image", format="png")
            stego_text = gr.Textbox(label="Stego text (generated by the language model)", lines=6)
            files = gr.File(label="Download stego.png and stego.txt", file_count="multiple")
        with gr.Tab("2. Extract"):
            with gr.Row():
                png_in = gr.File(label="Stego image (stego.png)", type="filepath")
                txt_in = gr.File(label="Stego text (stego.txt)", type="filepath")
            use_last = gr.Button("Use the files from the Hide tab")
            pasted = gr.Textbox(label="...or paste the stego text here", lines=4)
            with gr.Row():
                key2 = gr.Textbox(label="Shared key", value="demo-shared-key")
                method2 = gr.Radio(["hill", "lsbm", "lsbr"], value="hill", label="Image method")
            with gr.Row():
                jpeg = gr.Checkbox(label="Simulate attack: re-save image as JPEG")
                edit = gr.Checkbox(label="Simulate attack: change one word of the text")
            ext_btn = gr.Button("Extract secret", variant="primary")
            recovered = gr.Textbox(label="Recovered secret", lines=3)
            status = gr.Textbox(label="Receiver report", lines=4)

        hide_btn.click(hide, [photo, secret, key, method, max_tokens, size],
                       [cover_out, stego_out, stego_text, report, files, last_png, last_txt])
        use_last.click(lambda p, t: (p, t), [last_png, last_txt], [png_in, txt_in])
        ext_btn.click(extract, [png_in, txt_in, pasted, key2, method2, jpeg, edit], [recovered, status])
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--open", action="store_true", help="open the app in the default browser once it is ready")
    args = ap.parse_args()
    build(args.model).launch(server_name="127.0.0.1", server_port=args.port, inbrowser=args.open)
