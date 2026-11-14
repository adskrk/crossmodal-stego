"""Sender and receiver: one secret, two carriers.

Sender
  1. Decide the split. Default rule ("fill-text"): give the text channel as many
     secret bytes as fit in a natural-length passage (``max_text_tokens``), the
     image gets the rest. The size is found from the coder's own capacity
     measurement (usually 1-2 encodes); ``fill="exact"`` keeps the original
     bisection. A fixed ratio can be forced instead, for baselines.
  2. Build two sealed fragments (framing.split_message).
  3. Hide the text fragment with the LLM coder and the image fragment with the
     chosen image method.

Receiver
  Decode each carrier independently, then join_fragments() checks both CRCs and
  reports exactly which channel failed if anything is wrong.

What the receiver must share with the sender in advance: the key, the model id
and its version, the prompt list, the codec settings (top_k, C), the image
method. Nothing else travels with the carriers.
"""

from __future__ import annotations

import dataclasses
import random
import time
from typing import Optional

import numpy as np

from . import aont, image_channel, text_channel
from .framing import ReassemblyReport, join_fragments, split_message


@dataclasses.dataclass
class SendConfig:
    key: bytes
    model_path: str
    prompt: str
    image_method: str = "hill"
    C: float = 0.0
    top_k: int = 64
    max_text_tokens: int = 150
    text_guard: str = "filter"  # "filter" (subtext-codec) or "retry" (OD-Stega style); experiments found equal detectability, filter is cheaper
    text_share: Optional[float] = None  # None = fill-text rule; else fraction of the payload for the text
    fill: str = "fast"  # fill-text search: "fast" (sized from measured capacity, 1-3 encodes) or "exact" (bisection)
    aont: bool = True  # seal the secret in an all-or-nothing package before splitting (see aont.py)
    finish_sentence: bool = True  # let the text end at a sentence boundary after the payload
    # Line-break fix (see TextConfig.newline_fix). Measured on GPT-2 at C = 0 (192 messages, 96 test pairs): the
    # white-box likelihood detector went from AUC 0.76 to 0.60 and the style detector from 0.67 to 0.50.
    newline_fix: bool = True


@dataclasses.dataclass
class Carriers:
    text: str
    image: np.ndarray
    text_bytes: int    # payload bytes in the text (package bytes when aont=True)
    image_bytes: int   # payload bytes in the image
    text_stats: text_channel.TextStats
    image_quality: dict
    seconds: float
    payload_bytes: int = 0  # bytes actually split: the package, or the raw secret when aont=False


def _text_cfg(cfg: SendConfig, limit: Optional[int]) -> text_channel.TextConfig:
    return text_channel.TextConfig(model_path=cfg.model_path, prompt=cfg.prompt, top_k=cfg.top_k, C=cfg.C,
                                   max_new_tokens=limit, guard=cfg.text_guard,
                                   finish_sentence=cfg.finish_sentence, newline_fix=cfg.newline_fix)


def send(secret: bytes, cover: np.ndarray, cfg: SendConfig, model, tok, msg_id: Optional[int] = None) -> Carriers:
    t0 = time.time()
    msg_id = random.randrange(65536) if msg_id is None else msg_id
    tcfg = _text_cfg(cfg, cfg.max_text_tokens)
    payload = aont.pack(secret) if cfg.aont else secret
    lo, hi = aont.split_bounds(len(payload)) if cfg.aont else (0, len(payload))

    def encode_text(t_bytes, limit=True):
        tf, _ = split_message(payload, t_bytes, msg_id, cfg.key)
        c = tcfg if limit else dataclasses.replace(tcfg, max_new_tokens=None)
        return text_channel.encode(tf, c, model, tok, key=cfg.key)

    def try_text(t_bytes):
        try:
            return encode_text(t_bytes)
        except ValueError:  # did not fit within max_text_tokens
            return None

    if cfg.text_share is not None:
        best = min(max(int(round(cfg.text_share * len(payload))), lo), hi)
        result = encode_text(best, limit=False)
    elif cfg.fill == "exact":
        # Original fill-text search: the most payload bytes whose fragment fits in
        # max_text_tokens, by bisection over the byte count. Every probe is a real
        # encode, so this needs about log2(len(payload)) + 1 encodes.
        result = try_text(lo)
        best = lo
        if result is None:  # even the minimum share overflows: let the text run longer
            result = encode_text(lo, limit=False)
        else:
            a, b = lo, hi
            while a < b:
                mid = (a + b + 1) // 2
                r = try_text(mid)
                if r is not None:
                    a, best, result = mid, mid, r
                else:
                    b = mid - 1
    else:
        best, result = _fast_fill(payload, lo, hi, msg_id, cfg, tcfg, model, tok, encode_text)

    text, tstats = result
    _, imf = split_message(payload, best, msg_id, cfg.key)
    stego = image_channel.embed(cover, imf, cfg.key, cfg.image_method)
    return Carriers(text, stego, best, len(payload) - best, tstats,
                    image_channel.quality(cover, stego), time.time() - t0, len(payload))


def receive(text: Optional[str], image: Optional[np.ndarray], cfg: SendConfig, model, tok) -> ReassemblyReport:
    tf = imf = None
    try:
        if text is not None:
            tf = text_channel.decode(text, _text_cfg(cfg, None), model, tok, key=cfg.key)
    except ValueError:
        tf = None
    try:
        if image is not None:
            imf = image_channel.extract(image, cfg.key, cfg.image_method)
    except Exception:
        imf = None
    rep = join_fragments(tf, imf, cfg.key)
    if rep.ok and cfg.aont:
        try:
            rep.message = aont.unpack(rep.message)
        except ValueError as exc:
            rep.ok, rep.message, rep.reason = False, None, f"package did not open: {exc}"
    return rep


_MARGIN = 0.92        # aim this far below the measured capacity
_MAX_ATTEMPTS = 4


def _fast_fill(payload: bytes, lo: int, hi: int, msg_id: int, cfg: SendConfig, tcfg, model, tok, encode_text):
    """Fill-text split sized from measured capacity instead of bisection.

    Attempt 1 offers the most the text may take (``hi``). If it fits, that is
    the answer (one encode). If not, the coder reports how many payload bits it
    had placed when it ran out of tokens; that is this passage's measured
    capacity, and the next attempt offers ``_MARGIN`` of it. Usually 1 encode
    for short secrets and 2 for long ones, versus ~log2(n)+1 for bisection. If
    even the minimum share ``lo`` does not fit in ``max_text_tokens``, the text
    is allowed to run longer so that both carriers still hold their share.
    """
    base = text_channel.payload_bit_length(len(split_message(payload, 0, msg_id, cfg.key)[0]), cfg.key)

    def bytes_for(bit_budget: float) -> int:
        return max(0, min(hi, int((bit_budget - base) // 8)))

    size, margin = hi, _MARGIN
    for _ in range(_MAX_ATTEMPTS):
        try:
            return size, encode_text(size)
        except text_channel.CapacityExceeded as e:
            nxt = bytes_for(margin * e.bits_done)
            margin *= 0.9
        except ValueError:  # e.g. the retry guard found no clean tokenisation
            nxt = int(size * 0.8)
        if size <= lo:
            break
        size = max(lo, min(nxt, size - 1))
    return lo, encode_text(lo, limit=False)
