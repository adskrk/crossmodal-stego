"""End to end: one secret split across text + image, recovered exactly."""

import glob
import io
import os

import numpy as np
import pytest
from PIL import Image

from conftest import DEFAULT_LM, ROOT
from cmstego import system, text_channel
from cmstego.prompts import PROMPTS

pytestmark = pytest.mark.skipif(not (os.path.isdir(DEFAULT_LM) or "/" not in DEFAULT_LM),
                                reason="no language model available")


@pytest.fixture(scope="module")
def lm():
    return text_channel.load_lm(DEFAULT_LM)


def cover():
    p = sorted(glob.glob(os.path.join(ROOT, "data", "covers", "A_kodak", "*.png")))
    if p:
        return np.asarray(Image.open(p[0]).convert("RGB"))
    from skimage import data  # before setup.sh has fetched Kodak
    return np.ascontiguousarray(data.astronaut()[:256, :256, :3])


@pytest.mark.parametrize("method", ["lsbm", "hill"])
@pytest.mark.parametrize("secret", [b"", b"x", b"Meet at the library at 5pm." * 4, bytes(range(256)) * 8])
def test_fill_text_roundtrip(lm, method, secret):
    model, tok = lm
    cfg = system.SendConfig(key=b"k", model_path=DEFAULT_LM, prompt=PROMPTS[0], image_method=method,
                            max_text_tokens=120)
    car = system.send(secret, cover(), cfg, model, tok, msg_id=7)
    assert car.text_stats.tokens <= 120
    assert car.text_bytes + car.image_bytes == len(secret)
    rep = system.receive(car.text, car.image, cfg, model, tok)
    assert rep.ok and rep.message == secret


@pytest.mark.parametrize("share", [0.0, 0.5, 1.0])
def test_fixed_split_roundtrip(lm, share):
    model, tok = lm
    secret = b"fixed split test message!" * 2
    cfg = system.SendConfig(key=b"k", model_path=DEFAULT_LM, prompt=PROMPTS[2], image_method="hill",
                            text_share=share)
    car = system.send(secret, cover(), cfg, model, tok, msg_id=8)
    assert car.text_bytes == round(share * len(secret))
    rep = system.receive(car.text, car.image, cfg, model, tok)
    assert rep.ok and rep.message == secret


def test_jpeg_breaks_image_and_is_reported(lm):
    model, tok = lm
    cfg = system.SendConfig(key=b"k", model_path=DEFAULT_LM, prompt=PROMPTS[1], image_method="hill")
    car = system.send(b"robustness check " * 10, cover(), cfg, model, tok)
    buf = io.BytesIO()
    Image.fromarray(car.image).save(buf, format="JPEG", quality=90)
    damaged = np.asarray(Image.open(buf).convert("RGB"))
    rep = system.receive(car.text, damaged, cfg, model, tok)
    assert not rep.ok and rep.text_ok and not rep.image_ok


def test_wrong_key_fails(lm):
    model, tok = lm
    cfg = system.SendConfig(key=b"right", model_path=DEFAULT_LM, prompt=PROMPTS[1], image_method="lsbm")
    car = system.send(b"secret", cover(), cfg, model, tok)
    bad = system.SendConfig(key=b"wrong", model_path=DEFAULT_LM, prompt=PROMPTS[1], image_method="lsbm")
    assert not system.receive(car.text, car.image, bad, model, tok).ok
