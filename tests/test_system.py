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
    assert car.text_bytes + car.image_bytes == car.payload_bytes
    assert car.text_bytes >= 16 and car.image_bytes >= 16  # both carriers always hold a share
    rep = system.receive(car.text, car.image, cfg, model, tok)
    assert rep.ok and rep.message == secret


@pytest.mark.parametrize("share", [0.0, 0.5, 1.0])
def test_fixed_split_roundtrip(lm, share):
    model, tok = lm
    secret = b"fixed split test message!" * 2
    cfg = system.SendConfig(key=b"k", model_path=DEFAULT_LM, prompt=PROMPTS[2], image_method="hill",
                            text_share=share)
    car = system.send(secret, cover(), cfg, model, tok, msg_id=8)
    n = car.payload_bytes
    assert car.text_bytes == min(max(round(share * n), 16), n - 16)
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


@pytest.mark.parametrize("n", [5, 60, 3000])
def test_fast_fill_uses_few_encodes(lm, n, monkeypatch):
    model, tok = lm
    calls = {"n": 0}
    orig = text_channel._encode_once

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(text_channel, "_encode_once", counting)
    secret = (b"Meet at the library at 5pm. " * 200)[:n]
    cfg = system.SendConfig(key=b"k", model_path=DEFAULT_LM, prompt=PROMPTS[3], image_method="hill",
                            max_text_tokens=100)
    car = system.send(secret, cover(), cfg, model, tok, msg_id=9)
    assert calls["n"] <= 4
    assert car.text_bytes + car.image_bytes == car.payload_bytes
    rep = system.receive(car.text, car.image, cfg, model, tok)
    assert rep.ok and rep.message == secret


def test_exact_fill_still_available(lm):
    model, tok = lm
    secret = b"exact bisection path " * 3
    cfg = system.SendConfig(key=b"k", model_path=DEFAULT_LM, prompt=PROMPTS[0], image_method="lsbm",
                            max_text_tokens=100, fill="exact")
    car = system.send(secret, cover(), cfg, model, tok, msg_id=10)
    rep = system.receive(car.text, car.image, cfg, model, tok)
    assert rep.ok and rep.message == secret


@pytest.mark.parametrize("secret", [b"", b"x" * 40])
def test_without_aont_old_behaviour(lm, secret):
    model, tok = lm
    cfg = system.SendConfig(key=b"k", model_path=DEFAULT_LM, prompt=PROMPTS[0], image_method="lsbm",
                            max_text_tokens=120, aont=False)
    car = system.send(secret, cover(), cfg, model, tok, msg_id=11)
    assert car.payload_bytes == len(secret) and car.text_bytes + car.image_bytes == len(secret)
    rep = system.receive(car.text, car.image, cfg, model, tok)
    assert rep.ok and rep.message == secret


def test_one_carrier_alone_is_not_enough(lm):
    """With the package, even holding the key, one carrier's part opens nothing."""
    from cmstego import aont, image_channel
    from cmstego.framing import _unseal
    model, tok = lm
    secret = b"Meet at the library at 5pm."
    cfg = system.SendConfig(key=b"k", model_path=DEFAULT_LM, prompt=PROMPTS[1], image_method="hill")
    car = system.send(secret, cover(), cfg, model, tok, msg_id=12)
    img_part = _unseal(image_channel.extract(car.image, b"k", "hill"), b"k", "image").body
    assert secret not in img_part
    # pretend the text part were all zeros: the package must not open to the secret
    fake = bytes(car.text_bytes) + img_part
    try:
        assert aont.unpack(fake) != secret
    except ValueError:
        pass


@pytest.mark.parametrize("seed_prompt", [0, 4, 7])
def test_text_finishes_its_sentence(lm, seed_prompt):
    model, tok = lm
    secret = b"finish the sentence please"
    cfg = system.SendConfig(key=b"k", model_path=DEFAULT_LM, prompt=PROMPTS[seed_prompt], image_method="lsbm")
    car = system.send(secret, cover(), cfg, model, tok, msg_id=13)
    tail = car.text_stats.tail_tokens
    assert 0 <= tail <= 40
    if 0 < tail < 40:  # stopped because a sentence (or line) ended
        assert car.text.rstrip().endswith((".", "!", "?", '"', "'", ")")) or car.text.endswith("\n")
    rep = system.receive(car.text, car.image, cfg, model, tok)
    assert rep.ok and rep.message == secret


def test_finish_sentence_off_keeps_old_text_shape(lm):
    """With the option off (as in the recorded experiments) no tail is added."""
    model, tok = lm
    tcfg = text_channel.TextConfig(model_path=DEFAULT_LM, prompt=PROMPTS[0])
    text, st = text_channel.encode(b"abc", tcfg, model, tok, key=b"k", nonce=5)
    assert st.tail_tokens == 0
    assert text_channel.decode(text, tcfg, model, tok, key=b"k") == b"abc"


@pytest.mark.parametrize("prompt_id", [1, 6])
def test_newline_fix_roundtrip(lm, prompt_id):
    """Line breaks are shown doubled and halved back; the secret still comes back exactly."""
    model, tok = lm
    secret = b"line-break fix check " * 3
    cfg = system.SendConfig(key=b"k", model_path=DEFAULT_LM, prompt=PROMPTS[prompt_id], image_method="lsbm",
                            newline_fix=True)
    car = system.send(secret, cover(), cfg, model, tok, msg_id=14)
    body = car.text[len(PROMPTS[prompt_id]):]
    assert "\n" not in body.replace("\n\n", "")  # every line break is shown doubled
    rep = system.receive(car.text, car.image, cfg, model, tok)
    assert rep.ok and rep.message == secret
