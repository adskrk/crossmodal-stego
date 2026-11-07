"""Sender and receiver: one secret, two carriers.

Sender
  1. Decide the split. Default rule ("fill-text"): give the text channel as many
     secret bytes as fit in a natural-length passage (``max_text_tokens``), the
     image gets the rest. A fixed ratio can be forced instead, for baselines.
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

from . import image_channel, text_channel
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
    text_share: Optional[float] = None  # None = fill-text rule; else fraction of secret bytes


@dataclasses.dataclass
class Carriers:
    text: str
    image: np.ndarray
    text_bytes: int
    image_bytes: int
    text_stats: text_channel.TextStats
    image_quality: dict
    seconds: float


def _text_cfg(cfg: SendConfig, limit: Optional[int]) -> text_channel.TextConfig:
    return text_channel.TextConfig(model_path=cfg.model_path, prompt=cfg.prompt, top_k=cfg.top_k, C=cfg.C,
                                   max_new_tokens=limit, guard=cfg.text_guard)


def send(secret: bytes, cover: np.ndarray, cfg: SendConfig, model, tok, msg_id: Optional[int] = None) -> Carriers:
    t0 = time.time()
    msg_id = random.randrange(65536) if msg_id is None else msg_id
    tcfg = _text_cfg(cfg, cfg.max_text_tokens)

    def try_text(t_bytes):
        tf, _ = split_message(secret, t_bytes, msg_id, cfg.key)
        try:
            return text_channel.encode(tf, tcfg, model, tok, key=cfg.key)
        except ValueError:  # did not fit within max_text_tokens
            return None

    if cfg.text_share is not None:
        best = int(round(cfg.text_share * len(secret)))
        tcfg.max_new_tokens = None
        result = try_text(best)
        if result is None:
            raise ValueError("text fragment could not be encoded")
    else:
        # Fill-text rule: the most secret bytes whose fragment fits in
        # max_text_tokens. Every probe is a real encode (capacity depends on the
        # prompt and on chance), found by bisection over the byte count.
        lo, hi = 0, len(secret)
        result = try_text(0)
        if result is None:
            raise ValueError("even an empty text fragment does not fit; raise max_text_tokens")
        best = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            r = try_text(mid)
            if r is not None:
                lo, best, result = mid, mid, r
            else:
                hi = mid - 1

    text, tstats = result
    _, imf = split_message(secret, best, msg_id, cfg.key)
    stego = image_channel.embed(cover, imf, cfg.key, cfg.image_method)
    return Carriers(text, stego, best, len(secret) - best, tstats,
                    image_channel.quality(cover, stego), time.time() - t0)


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
    return join_fragments(tf, imf, cfg.key)
