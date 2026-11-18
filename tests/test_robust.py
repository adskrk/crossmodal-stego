"""Robust image method (DCT-QIM + Reed-Solomon) and format-preserving saving."""

import io

import numpy as np
import pytest
from PIL import Image

from cmstego import fileio
from cmstego import image_channel as ic
from test_image import covers


@pytest.fixture(scope="module")
def cover():
    return covers()[0]


def _jpeg(img, q):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=q)
    return np.asarray(Image.open(buf).convert("RGB"))


@pytest.mark.parametrize("n", [1, 50, 200])
def test_roundtrip(cover, n):
    data = bytes(range(n % 256)) * (n // 256 + 1)
    data = data[:n]
    assert ic.extract(ic.embed(cover, data, b"k", "robust"), b"k", "robust") == data


def test_capacity_reported_and_enforced(cover):
    cap = ic.robust_capacity_bytes(cover)
    assert cap > 100 and ic.capacity_bits(cover, "robust") == 8 * cap
    with pytest.raises(ValueError):
        ic.embed(cover, bytes(cap + 50), b"k", "robust")


@pytest.mark.parametrize("q", [95, 90])
def test_survives_jpeg(cover, q):
    data = b"survives recompression " * 4
    stego = ic.embed(cover, data, b"k", "robust")
    assert ic.extract(_jpeg(stego, q), b"k", "robust") == data


def test_survives_salt_and_pepper(cover):
    rng = np.random.default_rng(3)
    data = b"salt and pepper " * 4
    stego = ic.embed(cover, data, b"k", "robust").copy()
    hit = rng.random(stego.shape) < 0.01
    stego[hit] = rng.choice([0, 255], size=int(hit.sum())).astype(np.uint8)
    assert ic.extract(stego, b"k", "robust") == data


def test_wrong_key_fails(cover):
    stego = ic.embed(cover, b"secret", b"right", "robust")
    with pytest.raises(ValueError):
        ic.extract(stego, b"wrong", "robust")


def test_hill_breaks_under_jpeg_but_reports_it(cover):
    stego = ic.embed(cover, b"fragile", b"k", "hill")
    with pytest.raises(ValueError):
        ic.extract(_jpeg(stego, 95), b"k", "hill")


def test_save_like_keeps_alpha_and_metadata(tmp_path, cover):
    rgba = np.dstack([cover, np.full(cover.shape[:2], 255, np.uint8)])
    orig = tmp_path / "orig.png"
    from PIL.PngImagePlugin import PngInfo
    import struct
    meta = PngInfo()
    meta.add(b"gAMA", struct.pack(">I", 45455))  # PIL cannot write gamma itself
    Image.fromarray(rgba).save(orig, dpi=(300, 300), pnginfo=meta)
    stego = ic.embed(cover, b"keep the format", b"k", "hill")
    out = tmp_path / "stego.png"
    fileio.save_like(stego, str(orig), str(out))
    back = Image.open(out)
    assert back.mode == "RGBA"
    assert round(back.info["dpi"][0]) == 300 and "gamma" in back.info
    assert ic.extract(np.asarray(back.convert("RGB")), b"k", "hill") == b"keep the format"
