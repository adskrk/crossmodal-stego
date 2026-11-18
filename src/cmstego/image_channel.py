"""Image channel: hide bytes in an RGB cover image, saved losslessly (PNG).

Three methods, from weakest to strongest:

* ``lsbr`` — LSB replacement. Write message bits into the least significant
  bit of key-permuted colour values. Classic baseline; known to be detectable by
  structural attacks (RS, SPA) because it only ever moves 2k <-> 2k+1.
* ``lsbm`` — LSB matching (±1). Same positions, but when the LSB is wrong the
  value is moved up or down by 1 at random. Same distortion, no RS/SPA asymmetry.
* ``hill`` — content-adaptive: HILL costs (Li et al., ICIP 2014) say how
  expensive a ±1 change is at each colour value (cheap in texture, expensive in
  flat areas); syndrome-trellis codes (Filler et al., 2011, via pySTC, MIT)
  embed the message while near-minimising total cost. Only ~the changes that
  are needed are made, and mostly in busy regions.
* ``robust`` — prototype for channels that change the image (JPEG re-saving,
  noise, filtering). Bits go into low-frequency DCT coefficients of the
  brightness in JPEG's own 8x8 blocks, by dithered quantisation index
  modulation (QIM), at key-shuffled positions, protected by Reed-Solomon codes.
  Survives more damage, but makes larger changes: easier to detect.

All three treat the image as one long vector of H*W*3 colour values ("RGB,
each channel"), and frame the payload as length (4 bytes) + CRC32 (4 bytes) +
data so a damaged image fails loudly instead of returning wrong bytes.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from typing import Tuple

import numpy as np
from scipy import signal

FRAME = struct.Struct(">II")  # length, crc32
METHODS = ("lsbr", "lsbm", "hill", "robust")


def _seed(key: bytes, label: str) -> int:
    return int.from_bytes(hashlib.sha256(key + label.encode()).digest()[:8], "big")


def _frame(data: bytes) -> bytes:
    return FRAME.pack(len(data), zlib.crc32(data) & 0xFFFFFFFF) + data


def _unframe(blob: bytes) -> bytes:
    if len(blob) < FRAME.size:
        raise ValueError("image payload too short to hold its header")
    n, crc = FRAME.unpack_from(blob, 0)
    if n > len(blob) - FRAME.size:
        raise ValueError("image payload declares more bytes than were extracted (wrong key or damaged image)")
    data = blob[FRAME.size : FRAME.size + n]
    if zlib.crc32(data) & 0xFFFFFFFF != crc:
        raise ValueError("image payload failed its checksum (wrong key or damaged image)")
    return data


def _bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


HILL_MAX_RATE = 0.4  # bits per *usable* colour value
COST_CAP = 1e8  # HILL cost assigned to perfectly flat regions; such values are effectively "wet"
MIN_USABLE_FRACTION = 0.15  # below this the STC length header cannot be embedded reliably


def usable_values(cover: np.ndarray) -> int:
    """Colour values HILL + STC can really change: not stuck at 0/255 and not
    in a perfectly flat region (where HILL's cost hits COST_CAP)."""
    costs = np.stack([hill_costs(cover[..., k]) for k in range(3)], axis=-1)
    return int(np.count_nonzero((cover > 0) & (cover < 255) & (costs < COST_CAP / 10)))


def capacity_bits(cover: np.ndarray, method: str) -> int:
    """Payload bits this cover can carry with this method (excluding the 64-bit frame).

    For HILL + STC the limit is set by *usable* values. Saturated areas (blown-out
    sky) and perfectly flat areas (black borders) give STC no room, and above
    the limit it has no solution. Found by the image experiment on Kodak photo 20
    and the retina photo.
    """
    if method in ("lsbr", "lsbm"):
        return cover.size - 64
    if method == "robust":
        return 8 * robust_capacity_bytes(cover)
    usable = usable_values(cover)
    if usable < MIN_USABLE_FRACTION * cover.size:
        return 0
    usable_body = usable * (cover.size - STC_HEADER_REGION) / cover.size
    return max(0, int(HILL_MAX_RATE * usable_body) - 64)


# --------------------------------------------------------------------------
# LSB replacement / matching
# --------------------------------------------------------------------------


def _embed_lsb(cover: np.ndarray, data: bytes, key: bytes, matching: bool) -> np.ndarray:
    bits = _bits(_frame(data))
    flat = cover.reshape(-1).astype(np.int16)
    if len(bits) > flat.size:
        raise ValueError(f"payload needs {len(bits)} bits; image holds {flat.size}")
    order = np.random.default_rng(_seed(key, "pos")).permutation(flat.size)[: len(bits)]
    vals = flat[order]
    wrong = (vals & 1) != bits
    if not matching:
        vals = (vals & ~1) | bits
    else:
        step = np.random.default_rng(_seed(key, "pm")).choice([-1, 1], size=len(bits))
        step = np.where(vals == 0, 1, np.where(vals == 255, -1, step))
        vals = np.where(wrong, vals + step, vals)
    out = flat.copy()
    out[order] = vals
    return out.reshape(cover.shape).astype(np.uint8)


def _extract_lsb(stego: np.ndarray, key: bytes) -> bytes:
    flat = stego.reshape(-1)
    order = np.random.default_rng(_seed(key, "pos")).permutation(flat.size)
    head = (flat[order[:64]] & 1).astype(np.uint8)
    n, _ = FRAME.unpack(np.packbits(head).tobytes())
    total = 64 + 8 * n
    if total > flat.size:
        raise ValueError("declared length exceeds image capacity (wrong key or no payload)")
    bits = (flat[order[:total]] & 1).astype(np.uint8)
    return _unframe(np.packbits(bits).tobytes())


# --------------------------------------------------------------------------
# HILL + STC
# --------------------------------------------------------------------------

_KB = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float64)


def hill_costs(channel: np.ndarray) -> np.ndarray:
    """HILL embedding cost of a ±1 change at each pixel of one channel."""
    x = channel.astype(np.float64)
    r = np.abs(signal.convolve2d(x, _KB, mode="same", boundary="symm"))
    r = signal.convolve2d(r, np.ones((3, 3)) / 9.0, mode="same", boundary="symm")
    c = 1.0 / (r + 1e-10)
    c = signal.convolve2d(c, np.ones((15, 15)) / 225.0, mode="same", boundary="symm")
    return np.minimum(c, COST_CAP)


def _to_plane(img: np.ndarray) -> np.ndarray:
    """H x W x 3  ->  H x 3W (channels side by side) for pySTC's 2-D API."""
    return np.concatenate([img[..., k] for k in range(3)], axis=1)


def _from_plane(plane: np.ndarray, w: int) -> np.ndarray:
    return np.stack([plane[:, k * w : (k + 1) * w] for k in range(3)], axis=-1)


# pySTC's own hide() stores a 32-bit length in a fixed block of only 64 shuffled
# values; on heavily saturated covers that block cannot be embedded. We call its
# core STC routines directly with a 1024-value header region instead (rate 1/32).
STC_HEADER_REGION = 1024
_LEN = struct.Struct("!I")


def _stc_order(n: int, key: bytes) -> np.ndarray:
    return np.random.default_rng(_seed(key, "stc-order")).permutation(n)


def _embed_hill(cover: np.ndarray, data: bytes, key: bytes) -> np.ndarray:
    from pystc import pystc as core

    blob = _frame(data)
    if 8 * len(blob) > capacity_bits(cover, "hill") + 64:
        raise ValueError(f"payload exceeds this cover's HILL+STC capacity ({HILL_MAX_RATE} bits per usable value)")
    costs = np.stack([hill_costs(cover[..., k]) for k in range(3)], axis=-1)
    x = cover.reshape(-1).astype(np.int32)
    c = costs.reshape(-1).astype(np.float32)
    order = _stc_order(x.size, key)
    xs, cs = x[order], c[order]
    H = STC_HEADER_REGION
    head = core._hide_stc(xs[:H], cs[:H], cs[:H], core.bytes_to_bits(_LEN.pack(8 * len(blob))), mx=255, mn=0)
    body = core._hide_stc(xs[H:], cs[H:], cs[H:], core.bytes_to_bits(blob), mx=255, mn=0)
    out = np.empty_like(x)
    out[order] = np.concatenate([head, body])
    return out.reshape(cover.shape).astype(np.uint8)


def _extract_hill(stego: np.ndarray, key: bytes) -> bytes:
    from pystc import pystc as core

    x = stego.reshape(-1).astype(np.int32)
    xs = x[_stc_order(x.size, key)]
    H = STC_HEADER_REGION
    nbits = _LEN.unpack(core._unhide_stc(xs[:H], 32))[0]
    if nbits <= 0 or nbits > xs.size - H or nbits % 8:
        raise ValueError("implausible STC length (wrong key or damaged image)")
    return _unframe(bytes(core._unhide_stc(xs[H:], nbits)))


# --------------------------------------------------------------------------
# Robust prototype: DCT-domain QIM + Reed-Solomon
# --------------------------------------------------------------------------
#
# Brightness Y = 0.299 R + 0.587 G + 0.114 B is cut into JPEG's 8x8 blocks and
# transformed with the orthonormal 2-D DCT, which is the transform JPEG uses, so a
# JPEG quantisation step q on a coefficient is a step q in our units. Each bit
# moves one low-frequency coefficient onto one of two interleaved lattices
# (spacing ROBUST_DELTA, shifted by a key-derived dither); it survives as long as
# the coefficient moves by less than ROBUST_DELTA / 4. The change is applied
# equally to R, G and B, so colour is untouched. Bits are spread over key-shuffled
# positions and protected by Reed-Solomon codes (ROBUST_NSYM parity bytes per
# 255-byte block: up to ROBUST_NSYM / 2 wrong bytes per block are repaired).

ROBUST_DELTA = 28.0
ROBUST_COEFFS = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0))
ROBUST_NSYM = 64
ROBUST_HEADER_BITS = 16
ROBUST_HEADER_REPS = 15


def _luma(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float64)
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def _blocks(y: np.ndarray):
    hb, wb = y.shape[0] // 8, y.shape[1] // 8
    b = y[: hb * 8, : wb * 8].reshape(hb, 8, wb, 8).transpose(0, 2, 1, 3)
    return b, hb, wb


def _robust_slots(shape) -> int:
    return (shape[0] // 8) * (shape[1] // 8) * len(ROBUST_COEFFS)


def _robust_coeffs(img: np.ndarray) -> np.ndarray:
    from scipy.fft import dctn

    b, _, _ = _blocks(_luma(img))
    d = dctn(b, axes=(-2, -1), norm="ortho")
    return np.stack([d[:, :, u, v] for u, v in ROBUST_COEFFS], axis=-1).reshape(-1)


def _robust_key(n: int, key: bytes):
    order = np.random.default_rng(_seed(key, "qim-order")).permutation(n)
    dither = np.random.default_rng(_seed(key, "qim-dither")).uniform(0, ROBUST_DELTA, n)
    return order, dither


def _qim_embed(c: np.ndarray, bits: np.ndarray, dither: np.ndarray) -> np.ndarray:
    off = dither + bits * ROBUST_DELTA / 2
    return ROBUST_DELTA * np.round((c - off) / ROBUST_DELTA) + off


def _qim_detect(c: np.ndarray, dither: np.ndarray) -> np.ndarray:
    err = [np.abs(c - _qim_embed(c, np.full(c.shape, b), dither)) for b in (0, 1)]
    return (err[1] < err[0]).astype(np.uint8)


def _rs():
    from reedsolo import RSCodec

    return RSCodec(ROBUST_NSYM)


def robust_coded_bits(data: bytes) -> np.ndarray:
    """All bits the robust method writes: repeated length header + RS-coded frame."""
    body = bytes(_rs().encode(_frame(data)))
    head = np.unpackbits(np.frombuffer(struct.pack(">H", len(body)), np.uint8))
    return np.concatenate([np.repeat(head, ROBUST_HEADER_REPS), _bits(body)]).astype(np.uint8)


def robust_capacity_bytes(cover: np.ndarray) -> int:
    """Largest payload the robust method can carry in this cover."""
    coded = (_robust_slots(cover.shape) - ROBUST_HEADER_BITS * ROBUST_HEADER_REPS) // 8
    blocks = -(-coded // 255)
    return max(0, coded - blocks * ROBUST_NSYM - FRAME.size)


def _embed_robust(cover: np.ndarray, data: bytes, key: bytes) -> np.ndarray:
    from scipy.fft import idctn

    bits = robust_coded_bits(data)
    n = _robust_slots(cover.shape)
    if len(bits) > n:
        raise ValueError(f"payload needs {len(bits)} robust slots; this cover has {n} "
                         f"(at most {robust_capacity_bytes(cover)} bytes)")
    order, dither = _robust_key(n, key)
    pos, dit = order[: len(bits)], dither[order[: len(bits)]]
    out = cover.astype(np.float64)
    for _ in range(3):  # re-quantise after rounding and clipping
        c = _robust_coeffs(np.clip(np.round(out), 0, 255).astype(np.uint8))
        delta = np.zeros_like(c)
        delta[pos] = _qim_embed(c[pos], bits, dit) - c[pos]
        _, hb, wb = _blocks(out[..., 0])
        dc = np.zeros((hb, wb, 8, 8))
        delta = delta.reshape(hb, wb, len(ROBUST_COEFFS))
        for k, (u, v) in enumerate(ROBUST_COEFFS):
            dc[:, :, u, v] = delta[:, :, k]
        dy = idctn(dc, axes=(-2, -1), norm="ortho").transpose(0, 2, 1, 3).reshape(hb * 8, wb * 8)
        out[: hb * 8, : wb * 8] += dy[..., None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def robust_raw_bits(stego: np.ndarray, key: bytes, nbits: int, repair: bool = False) -> np.ndarray:
    """The first ``nbits`` written bits as read back (before Reed-Solomon correction)."""
    c = _robust_coeffs(repair_impulses(stego) if repair else stego)
    order, dither = _robust_key(c.size, key)
    pos = order[:nbits]
    return _qim_detect(c[pos], dither[pos])


def repair_impulses(img: np.ndarray, threshold: int = 40) -> np.ndarray:
    """Undo salt & pepper damage before decoding: a value of exactly 0 or 255 that
    differs from its 3x3 median by more than ``threshold`` is replaced by that
    median. Genuinely black or white areas match their median and are left alone."""
    from scipy import ndimage

    out = img.copy()
    for k in range(img.shape[2]):
        ch = img[..., k]
        med = ndimage.median_filter(ch, size=3)
        bad = ((ch == 0) | (ch == 255)) & (np.abs(ch.astype(np.int16) - med) > threshold)
        out[..., k][bad] = med[bad]
    return out


def _extract_robust(stego: np.ndarray, key: bytes) -> bytes:
    """Decode as received; if that fails, repair salt & pepper damage and try again."""
    try:
        return _extract_robust_once(stego, key)
    except ValueError:
        return _extract_robust_once(repair_impulses(stego), key)


def _extract_robust_once(stego: np.ndarray, key: bytes) -> bytes:
    from reedsolo import ReedSolomonError

    n = _robust_slots(stego.shape)
    hb = ROBUST_HEADER_BITS * ROBUST_HEADER_REPS
    if n < hb:
        raise ValueError("image too small for the robust method")
    head = robust_raw_bits(stego, key, hb, repair=False).reshape(ROBUST_HEADER_BITS, ROBUST_HEADER_REPS)
    length = int.from_bytes(np.packbits((head.sum(1) * 2 > ROBUST_HEADER_REPS).astype(np.uint8)).tobytes(), "big")
    if length == 0 or hb + 8 * length > n:
        raise ValueError("implausible robust length (wrong key or damaged image)")
    raw = robust_raw_bits(stego, key, hb + 8 * length, repair=False)[hb:]
    try:
        frame = bytes(_rs().decode(np.packbits(raw).tobytes())[0])
    except ReedSolomonError as exc:
        raise ValueError(f"too many errors to repair ({exc})") from exc
    return _unframe(frame)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def embed(cover: np.ndarray, data: bytes, key: bytes, method: str = "hill", verify: bool = True) -> np.ndarray:
    """Hide ``data`` in ``cover``. With ``verify`` (default), the stego image is
    decoded before it is returned, so a failed embedding raises instead of
    producing an image that cannot be read."""
    if cover.dtype != np.uint8 or cover.ndim != 3 or cover.shape[2] != 3:
        raise ValueError("cover must be an H x W x 3 uint8 RGB array")
    if method == "lsbr":
        stego = _embed_lsb(cover, data, key, matching=False)
    elif method == "lsbm":
        stego = _embed_lsb(cover, data, key, matching=True)
    elif method == "hill":
        stego = _embed_hill(cover, data, key)
    elif method == "robust":
        stego = _embed_robust(cover, data, key)
    else:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if verify:
        try:
            ok = extract(stego, key, method) == data
        except ValueError:
            ok = False
        if not ok:
            raise ValueError(f"{method} embedding did not verify on this cover; choose another cover or a smaller payload")
    return stego


def extract(stego: np.ndarray, key: bytes, method: str = "hill") -> bytes:
    if method in ("lsbr", "lsbm"):
        return _extract_lsb(stego, key)
    if method == "hill":
        return _extract_hill(stego, key)
    if method == "robust":
        return _extract_robust(stego, key)
    raise ValueError(f"unknown method {method!r}")


def quality(cover: np.ndarray, stego: np.ndarray) -> dict:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    changed = float((cover != stego).mean())
    psnr = float("inf") if changed == 0 else float(peak_signal_noise_ratio(cover, stego, data_range=255))
    ssim = float(structural_similarity(cover, stego, channel_axis=2, data_range=255))
    return {"psnr_db": psnr, "ssim": ssim, "change_rate": changed}
