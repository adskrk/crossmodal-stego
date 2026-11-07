"""Message framing: turning a secret into two self-checking fragments and back.

Each channel carries one *fragment*:

    header (10 bytes) = msg_id (2) | offset (2) | total_len (2) | msg_crc32 (4)
    body              = the slice secret[offset : offset + len(body)]

The channel codecs add their own length + CRC32 around whatever bytes they
carry, so a damaged channel is caught there. ``msg_crc32`` is a second check
over the *reassembled* message, so a wrong pairing of fragments (e.g. text from
one message, image from another) is also caught.

Optional *whitening* XORs every fragment with a keystream derived from a shared
key (SHAKE-256). Arithmetic-coding steganography assumes uniformly random input
bits; raw text is not uniform, so whitening is on by default. It is an ablation
switch, not a security claim.
"""

from __future__ import annotations

import dataclasses
import hashlib
import struct
import zlib
from typing import List, Optional, Sequence, Tuple

HEADER = struct.Struct(">HHHI")  # msg_id, offset, total_len, msg_crc32
HEADER_BYTES = HEADER.size  # 10
MAX_MESSAGE_BYTES = 0xFFFF


@dataclasses.dataclass(frozen=True)
class Fragment:
    msg_id: int
    offset: int
    total_len: int
    msg_crc32: int
    body: bytes

    def to_bytes(self) -> bytes:
        return HEADER.pack(self.msg_id, self.offset, self.total_len, self.msg_crc32) + self.body

    @classmethod
    def from_bytes(cls, data: bytes) -> "Fragment":
        if len(data) < HEADER_BYTES:
            raise ValueError("fragment shorter than its header")
        msg_id, offset, total_len, crc = HEADER.unpack_from(data, 0)
        return cls(msg_id, offset, total_len, crc, bytes(data[HEADER_BYTES:]))


def keystream(key: bytes, msg_id: int, channel: str, length: int) -> bytes:
    """Deterministic pseudo-random bytes for whitening one channel's fragment."""
    seed = key + struct.pack(">H", msg_id) + channel.encode()
    return hashlib.shake_256(seed).digest(length)


def xor_bytes(data: bytes, stream: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(data, stream))


def whiten(data: bytes, key: Optional[bytes], msg_id: int, channel: str) -> bytes:
    """XOR with a keystream; applying it twice returns the input (involution)."""
    if not key:
        return data
    return xor_bytes(data, keystream(key, msg_id, channel, len(data)))


# The msg_id itself must be readable before un-whitening, so it is whitened with
# a keystream that does not depend on it (msg_id = 0 slot).
def _seal(frag: Fragment, key: Optional[bytes], channel: str) -> bytes:
    raw = frag.to_bytes()
    head_id, rest = raw[:2], raw[2:]
    head_id = whiten(head_id, key, 0, channel + "/id")
    rest = whiten(rest, key, frag.msg_id, channel)
    return head_id + rest


def _unseal(data: bytes, key: Optional[bytes], channel: str) -> Fragment:
    if len(data) < HEADER_BYTES:
        raise ValueError("fragment shorter than its header")
    head_id = whiten(data[:2], key, 0, channel + "/id")
    msg_id = struct.unpack(">H", head_id)[0]
    rest = whiten(data[2:], key, msg_id, channel)
    return Fragment.from_bytes(head_id + rest)


def split_message(
    secret: bytes,
    text_bytes: int,
    msg_id: int,
    key: Optional[bytes] = None,
) -> Tuple[bytes, bytes]:
    """Return (text_fragment, image_fragment) as sealed byte strings.

    ``text_bytes`` is how many *secret* bytes go to the text channel (0..len).
    The image channel gets the rest. Both fragments always exist, even if their
    body is empty, so the receiver always knows the total length and CRC.
    """
    if len(secret) > MAX_MESSAGE_BYTES:
        raise ValueError(f"secret is {len(secret)} bytes; maximum is {MAX_MESSAGE_BYTES}")
    if not 0 <= text_bytes <= len(secret):
        raise ValueError("text_bytes must be between 0 and len(secret)")
    crc = zlib.crc32(secret) & 0xFFFFFFFF
    t = Fragment(msg_id, 0, len(secret), crc, secret[:text_bytes])
    i = Fragment(msg_id, text_bytes, len(secret), crc, secret[text_bytes:])
    return _seal(t, key, "text"), _seal(i, key, "image")


@dataclasses.dataclass
class ReassemblyReport:
    ok: bool
    message: Optional[bytes]
    reason: str
    text_ok: bool
    image_ok: bool


def join_fragments(
    text_frag: Optional[bytes],
    image_frag: Optional[bytes],
    key: Optional[bytes] = None,
) -> ReassemblyReport:
    """Rebuild the secret from the two sealed fragments and verify it."""
    frags: List[Fragment] = []
    text_ok = image_ok = False
    reasons = []
    for name, blob in (("text", text_frag), ("image", image_frag)):
        if blob is None:
            reasons.append(f"{name} fragment missing")
            continue
        try:
            f = _unseal(blob, key, name)
            frags.append(f)
            if name == "text":
                text_ok = True
            else:
                image_ok = True
        except ValueError as exc:
            reasons.append(f"{name} fragment unreadable: {exc}")

    if len(frags) != 2:
        return ReassemblyReport(False, None, "; ".join(reasons), text_ok, image_ok)

    a, b = sorted(frags, key=lambda f: f.offset)
    if a.msg_id != b.msg_id or a.total_len != b.total_len or a.msg_crc32 != b.msg_crc32:
        return ReassemblyReport(False, None, "fragments belong to different messages", text_ok, image_ok)
    if a.offset != 0 or a.offset + len(a.body) != b.offset or b.offset + len(b.body) != a.total_len:
        return ReassemblyReport(False, None, "fragment offsets do not tile the message", text_ok, image_ok)
    msg = a.body + b.body
    if zlib.crc32(msg) & 0xFFFFFFFF != a.msg_crc32:
        return ReassemblyReport(False, None, "message CRC mismatch", text_ok, image_ok)
    return ReassemblyReport(True, msg, "ok", text_ok, image_ok)


def bytes_to_bits(data: bytes) -> List[int]:
    return [(b >> (7 - i)) & 1 for b in data for i in range(8)]


def bits_to_bytes(bits: Sequence[int]) -> bytes:
    if len(bits) % 8:
        raise ValueError("bit count must be a multiple of 8")
    out = bytearray()
    for i in range(0, len(bits), 8):
        v = 0
        for b in bits[i : i + 8]:
            v = (v << 1) | int(b)
        out.append(v)
    return bytes(out)
