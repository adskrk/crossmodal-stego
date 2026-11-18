import glob
import os

import numpy as np
import pytest
from PIL import Image

from cmstego import image_channel as ic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def covers(n=3):
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "covers", "A_kodak", "*.png")))
    if not paths:  # before setup.sh has fetched Kodak: use photos bundled with scikit-image
        from skimage import data
        imgs = [data.astronaut(), data.coffee(), data.chelsea()]
        return [np.ascontiguousarray(im[:256, :256, :3]) for im in imgs[:n]]
    return [np.asarray(Image.open(p).convert("RGB")) for p in paths[:n]]


PM1_METHODS = ("lsbr", "lsbm", "hill")  # ±1 methods; the robust method has its own tests (test_robust.py)


@pytest.mark.parametrize("method", PM1_METHODS)
@pytest.mark.parametrize("nbytes", [0, 1, 100, 2000])
def test_roundtrip(method, nbytes):
    rng = np.random.default_rng(nbytes)
    for cover in covers():
        data = rng.integers(0, 256, nbytes, dtype=np.uint8).tobytes()
        stego = ic.embed(cover, data, b"key", method)
        assert stego.shape == cover.shape and stego.dtype == np.uint8
        assert int(np.abs(stego.astype(int) - cover.astype(int)).max()) <= 1  # only ±1 changes
        assert ic.extract(stego, b"key", method) == data


@pytest.mark.parametrize("method", ic.METHODS)
def test_png_save_load_roundtrip(method, tmp_path):
    cover = covers(1)[0]
    data = b"survives a lossless save" * 10
    stego = ic.embed(cover, data, b"key", method)
    p = tmp_path / "s.png"
    Image.fromarray(stego).save(p)
    assert ic.extract(np.asarray(Image.open(p)), b"key", method) == data


@pytest.mark.parametrize("method", ic.METHODS)
def test_wrong_key_does_not_return_wrong_bytes(method):
    cover = covers(1)[0]
    stego = ic.embed(cover, b"secret bytes" * 20, b"right", method)
    with pytest.raises(Exception):
        out = ic.extract(stego, b"wrong", method)
        assert out != b"secret bytes" * 20


def test_psnr_matches_arithmetic_for_full_rate_lsbr():
    # Full-rate random LSB replacement: about half of all values change by 1,
    # so MSE ~ 0.5 and PSNR ~ 10*log10(255^2/0.5) ~ 51.1 dB.
    cover = covers(1)[0]
    n = cover.size // 8 - 8
    data = np.random.default_rng(0).integers(0, 256, n, dtype=np.uint8).tobytes()
    q = ic.quality(cover, ic.embed(cover, data, b"k", "lsbr"))
    assert abs(q["psnr_db"] - 51.14) < 0.5


def test_flat_or_saturated_cover_refuses_instead_of_breaking():
    # Regression test for an earlier bug: a cover that is mostly blown-out or
    # perfectly flat has almost no room for HILL + STC. It must refuse loudly.
    rng = np.random.default_rng(3)
    cover = np.zeros((128, 128, 3), dtype=np.uint8)
    cover[:, :12] = rng.integers(1, 255, (128, 12, 3))  # small textured strip
    assert ic.capacity_bits(cover, "hill") == 0
    with pytest.raises(ValueError):
        ic.embed(cover, b"x" * 50, b"k", "hill")


def test_capacity_is_respected_on_real_covers():
    for cover in covers(3):
        cap = ic.capacity_bits(cover, "hill")
        n = max(1, cap // 8 - 8)
        data = np.random.default_rng(4).integers(0, 256, n, dtype=np.uint8).tobytes()
        assert ic.extract(ic.embed(cover, data, b"k", "hill"), b"k", "hill") == data


def test_hill_changes_fewer_values_than_lsbm():
    cover = covers(1)[0]
    data = np.random.default_rng(1).integers(0, 256, 3000, dtype=np.uint8).tobytes()
    lm = ic.quality(cover, ic.embed(cover, data, b"k", "lsbm"))["change_rate"]
    hi = ic.quality(cover, ic.embed(cover, data, b"k", "hill"))["change_rate"]
    assert hi < lm
