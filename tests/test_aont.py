"""All-or-nothing package: round trips, sizes, and one part alone revealing nothing."""

import os

import pytest

from cmstego import aont


@pytest.mark.parametrize("n", [0, 1, 13, 14, 15, 100, 5000])
def test_roundtrip_and_length(n):
    secret = os.urandom(n)
    p = aont.pack(secret)
    assert len(p) == aont.package_length(n) >= 2 * aont.MIN_SHARE
    assert aont.unpack(p) == secret


def test_fresh_key_every_time():
    assert aont.pack(b"same secret") != aont.pack(b"same secret")


def test_package_does_not_contain_plaintext():
    secret = b"Meet at the library at 5pm. " * 4
    assert b"library" not in aont.pack(secret)


@pytest.mark.parametrize("pos", [0, 5, -1, -16, -17])
def test_any_changed_byte_breaks_everything(pos):
    """A part that differs anywhere (even one byte) gives an unrelated output."""
    secret = b"x" * 60
    p = bytearray(aont.pack(secret))
    p[pos] ^= 0x01
    try:
        out = aont.unpack(bytes(p))
    except ValueError:
        return
    assert out != secret
    # not just a one-byte difference: the whole content is scrambled
    assert sum(a == b for a, b in zip(out, secret)) < len(secret) // 2


def test_split_bounds():
    lo, hi = aont.split_bounds(aont.package_length(5))
    assert lo == aont.MIN_SHARE and hi == aont.package_length(5) - aont.MIN_SHARE and lo <= hi


def test_too_long():
    with pytest.raises(ValueError):
        aont.pack(bytes(aont.MAX_SECRET_BYTES + 1))
