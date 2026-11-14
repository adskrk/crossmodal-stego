"""All-or-nothing package ("shredded lock-box") for the cross-modal split.

The secret is turned into one package that can only be opened when every byte
of it is present, and only then is the package cut between the two carriers.
Whoever holds just one carrier's part learns nothing about the secret, however
the cut is placed, so the split ratio becomes a free parameter that a policy
(e.g. RL) can choose for low detectability.

Construction (a package transform in the sense of Rivest, "All-or-nothing
encryption and the package transform", FSE 1997; here with SHAKE-256 / SHA-256
in place of a block cipher):

    k  = 16 random bytes                        (fresh for every message)
    m' = len(secret) (2 bytes) || secret || random padding
    c  = m' XOR SHAKE-256("cmstego-aont-enc" || k)
    t  = k XOR SHA-256("cmstego-aont-mask" || c)[:16]
    package = c || t

Opening: k = t XOR SHA-256(... || c)[:16], then m' = c XOR SHAKE-256(... || k).
Without all of c the hash, hence k, is unknown; without all of t, k is unknown.

Each carrier must still hold enough unknown bytes that the other side cannot
guess them: ``MIN_SHARE`` (16 bytes = 128 bits) per carrier. Short secrets are
padded so the package is at least 2 * MIN_SHARE bytes.

Scope: this protects against someone who obtains one carrier's decoded part,
e.g. if the shared key leaks later or one carrier is captured. It does not
replace the shared key, which is still what finds and un-whitens each part.
SHAKE-256 and SHA-256 are standard primitives, but this construction has not
had an independent security review.
"""

from __future__ import annotations

import hashlib
import os
import struct
from typing import Callable

KEY_BYTES = 16
MIN_SHARE = 16
_LEN = struct.Struct(">H")
MAX_SECRET_BYTES = 0xFFFF - KEY_BYTES - _LEN.size  # package must fit framing's 16-bit length
_ENC = b"cmstego-aont-enc"
_MASK = b"cmstego-aont-mask"


def _xor(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


def _stream(k: bytes, n: int) -> bytes:
    return hashlib.shake_256(_ENC + k).digest(n)


def _mask(c: bytes) -> bytes:
    return hashlib.sha256(_MASK + c).digest()[:KEY_BYTES]


def package_length(secret_len: int) -> int:
    """Length of the package for a secret of ``secret_len`` bytes."""
    return max(2 * MIN_SHARE, _LEN.size + secret_len + KEY_BYTES)


def pack(secret: bytes, rng: Callable[[int], bytes] = os.urandom) -> bytes:
    """Seal ``secret`` into an all-or-nothing package."""
    if len(secret) > MAX_SECRET_BYTES:
        raise ValueError(f"secret is {len(secret)} bytes; maximum is {MAX_SECRET_BYTES}")
    k = rng(KEY_BYTES)
    body = _LEN.pack(len(secret)) + secret
    body += rng(package_length(len(secret)) - KEY_BYTES - len(body))
    c = _xor(body, _stream(k, len(body)))
    return c + _xor(k, _mask(c))


def unpack(package: bytes) -> bytes:
    """Open a complete package. Raises ValueError if it is malformed."""
    if len(package) < KEY_BYTES + _LEN.size:
        raise ValueError("package too short")
    c, t = package[:-KEY_BYTES], package[-KEY_BYTES:]
    k = _xor(t, _mask(c))
    body = _xor(c, _stream(k, len(c)))
    (n,) = _LEN.unpack_from(body, 0)
    if n > len(body) - _LEN.size:
        raise ValueError("package does not open (declared length too large)")
    return body[_LEN.size:_LEN.size + n]


def split_bounds(package_len: int) -> tuple:
    """Allowed range for the number of package bytes given to the text carrier."""
    return MIN_SHARE, package_len - MIN_SHARE
