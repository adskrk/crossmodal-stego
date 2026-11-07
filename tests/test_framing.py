import os
import random

import pytest

from cmstego.framing import join_fragments, split_message


@pytest.mark.parametrize("key", [None, b"shared-key"])
def test_split_join_roundtrip_all_splits(key):
    rng = random.Random(1)
    for n in [0, 1, 7, 64, 300]:
        secret = bytes(rng.randrange(256) for _ in range(n))
        for t in sorted({0, n // 3, n // 2, n}):
            tf, imf = split_message(secret, t, msg_id=rng.randrange(65536), key=key)
            rep = join_fragments(tf, imf, key=key)
            assert rep.ok and rep.message == secret, (n, t, rep.reason)


def test_whitening_changes_bytes_and_wrong_key_fails():
    secret = b"attack at dawn" * 4
    tf, imf = split_message(secret, 20, msg_id=42, key=b"k1")
    plain_tf, _ = split_message(secret, 20, msg_id=42, key=None)
    assert tf != plain_tf
    assert not join_fragments(tf, imf, key=b"k2").ok


def test_mismatched_fragments_detected():
    a = split_message(b"message one", 5, msg_id=1, key=b"k")
    b = split_message(b"message two", 5, msg_id=2, key=b"k")
    rep = join_fragments(a[0], b[1], key=b"k")
    assert not rep.ok


def test_missing_channel_reported():
    tf, _ = split_message(b"hello", 2, msg_id=3, key=b"k")
    rep = join_fragments(tf, None, key=b"k")
    assert not rep.ok and rep.text_ok and not rep.image_ok
