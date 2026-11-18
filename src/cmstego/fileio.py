"""Save a stego image so that it looks like the file it came from.

A stego PNG written from scratch drops the original's transparency channel and
its metadata chunks (colour profile, gamma, resolution, text, EXIF). Those
differences say nothing about hidden data, but they do show the file was
re-written. ``save_like`` keeps them:

* if the original has an alpha channel, the stego RGB is recombined with the
  original alpha (hiding only ever touches R, G and B);
* every ancillary PNG chunk of the original (sRGB, gAMA, cHRM, iCCP, pHYs, tEXt,
  iTXt, zTXt, eXIf, tIME, ...) is copied into the new file, before the image data
  as the PNG standard requires for most of them.

Only PNG originals are matched chunk by chunk. A JPEG original cannot carry the
lossless methods' data (JPEG would destroy it), so their stego is saved as PNG.
"""

from __future__ import annotations

import io
import struct
import zlib
from typing import List, Tuple

import numpy as np
from PIL import Image

_SIG = b"\x89PNG\r\n\x1a\n"
_CRITICAL = {b"IHDR", b"PLTE", b"IDAT", b"IEND"}


def _chunks(data: bytes) -> List[Tuple[bytes, bytes]]:
    if not data.startswith(_SIG):
        raise ValueError("not a PNG file")
    out, i = [], len(_SIG)
    while i + 8 <= len(data):
        n, kind = struct.unpack(">I4s", data[i:i + 8])
        out.append((kind, data[i + 8:i + 8 + n]))
        i += 12 + n
    return out


def _chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def save_like(stego_rgb: np.ndarray, original_path: str, out_path: str) -> None:
    """Write ``stego_rgb`` as a PNG that keeps the original's alpha and metadata."""
    orig = Image.open(original_path)
    img = Image.fromarray(stego_rgb)
    same_size = orig.size == img.size
    if orig.mode in ("RGBA", "LA", "PA") and same_size:
        alpha = orig.convert("RGBA").getchannel("A")
        img = Image.merge("RGBA", (*img.split(), alpha))
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=9)
    new = _chunks(buf.getvalue())
    try:
        with open(original_path, "rb") as f:
            extra = [(k, b) for k, b in _chunks(f.read()) if k not in _CRITICAL]
    except ValueError:  # not a PNG original: nothing chunk-level to copy
        extra = []
    if not same_size:  # resized: resolution/time chunks still fine; drop nothing else
        pass
    have = {k for k, _ in new}
    extra = [(k, b) for k, b in extra if k not in have]
    out = [_SIG]
    for kind, body in new:
        if kind == b"IDAT" and extra:
            out += [_chunk(k, b) for k, b in extra]
            extra = []
        out.append(_chunk(kind, body))
    with open(out_path, "wb") as f:
        f.write(b"".join(out))
