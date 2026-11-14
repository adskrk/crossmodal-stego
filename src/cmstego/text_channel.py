"""Text channel: arithmetic-coding steganography over a causal language model.

Built on subtext-codec (MIT, https://github.com/shevisj/subtext-codec). We reuse
its arithmetic coder, payload framing (length + CRC32), KV-cached stepper,
rolling context window and tokenizer-stability filter. We change three things:

1. OD-Stega budget (Huang et al., EACL 2026). Per step:
       P      = model's full next-token distribution (temperature 1)
       P_hat  = P restricted to the candidate tokens (top-k, maybe filtered)
       delta  = C * H(P);   delta' = max(0, delta - KL(P_hat || P))
       Q      = P_hat ** (1/T) renormalised, T >= 1 so that KL(Q || P_hat) = delta'
   OD-Stega's Theorem 1: this temperature form maximises entropy under the KL
   budget. T is found by bisection. C = 0 gives T = 1.

2. Nonce + whitening of the whole bitstream (experimental finding). subtext-codec sends
   its length field in the clear, so equal-length messages start with identical
   bits and therefore identical opening tokens — a detector spots this almost
   immediately (held-out AUC about 0.95 in our first run). With a key, the
   bitstream becomes:  nonce (32 random bits) || XOR(length || crc || data,
   SHAKE-256(key || nonce)).  Every message now starts differently.

3. Two tokenisation guards (the receiver re-tokenises the text, so the
   encoder's token ids must survive a write-out / read-back):
     "filter": subtext-codec's guard — only offer tokens that survive. It
               works, but stego-text can then *never* contain those tokens while
               ordinary samples sometimes do: a one-sided fingerprint (in our
               first-stage results, 44% of ordinary 90-token samples contained one).
     "retry":  OD-Stega's approach — keep the full top-k distribution and, if
               the finished text does not re-tokenise to the same ids, pick a new
               nonce and encode again. Needs a key (the nonce is what changes).

We record, per emitted token, the exact KL of the integer table the coder used
against (a) the model's full distribution and (b) ordinary top-k sampling, which
is what an innocent user of the same model (and our cover texts) would produce.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
from typing import List, Optional, Sequence, Tuple

import torch

from subtext_codec import set_deterministic
from subtext_codec.arithmetic import (
    ENCODER_TAIL,
    FREQ_TOTAL,
    ArithmeticDecoder,
    ArithmeticEncoder,
    cumulative,
    quantize_frequencies,
)
from subtext_codec.codec import (
    MIN_ALPHABET,
    _STALL_LIMIT,
    _declared_length,
    _frame_payload,
    _special_ids,
    _stable_candidates,
    _Stepper,
    _token_ids,
    _unframe_payload,
)

LN2 = math.log(2.0)
NONCE_BITS = 32
MAX_RETRIES = 64


@dataclasses.dataclass
class TextConfig:
    model_path: str
    prompt: str
    top_k: int = 64
    C: float = 0.0  # OD-Stega budget constant; 0 = no distribution change
    max_new_tokens: Optional[int] = 2000
    device: str = "cpu"
    guard: str = "filter"  # "filter" (subtext-codec) or "retry" (OD-Stega style; needs a key)
    # After the hidden bits, keep writing with ordinary sampling until a sentence
    # ends, so the text does not stop mid-sentence. The decoder stops reading once
    # it has the payload, so these tokens carry nothing and are ignored. Off by
    # default so the recorded experiments stay reproducible; system.send turns it on.
    finish_sentence: bool = False
    max_tail_tokens: int = 40
    # Line-break fingerprint fix. Ordinary GPT-2 text often contains the single
    # token "\n\n" (a blank line); stego-text almost never does, because the
    # word after it re-reads as "\n", "\n", word and the stability filter
    # rejects that. With the fix, tokens containing "\n\n" are left out of the
    # alphabet, every line break of the stego-text is shown doubled, and the
    # receiver halves them again before decoding. Off by default (experiments).
    newline_fix: bool = False


@dataclasses.dataclass
class TextStats:
    tokens: int
    payload_bytes: int
    bits_per_token: float
    kl_bits_total: float  # sum over tokens of KL(q_used || P), in bits
    kl_bits_per_token: float
    kl_vs_topk_bits_per_token: float  # KL vs ordinary top-k sampling (the cover distribution)
    mean_entropy_bits: float  # mean H(P) of the model at each step
    mean_temperature: float
    attempts: int = 1  # encodes needed (retry guard)
    tail_tokens: int = 0  # ordinary tokens appended after the payload (finish_sentence)


def context_limit(model) -> Optional[int]:
    """The model's maximum sequence length (GPT-2: 1024)."""
    cfg = model.config
    for name in ("n_positions", "max_position_embeddings"):
        v = getattr(cfg, name, None)
        if isinstance(v, int) and v > 0:
            return v
    return None


def load_lm(model_path: str, device: str = "cpu"):
    """Load a causal LM in fp32 with deterministic settings."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_deterministic(0)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model.to(device).eval()
    return model, tok


# --------------------------------------------------------------------------
# bit helpers: nonce + keystream whitening
# --------------------------------------------------------------------------


def _int_bits(v: int, n: int) -> List[int]:
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]


def _bits_int(bits: Sequence[int]) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


class _Keystream:
    """Bits of SHAKE-256(key || 'text' || nonce), extended on demand (prefix-stable)."""

    def __init__(self, key: bytes, nonce: int):
        self.seed = key + b"text" + nonce.to_bytes(4, "big")
        self.bits: List[int] = []

    def get(self, n: int) -> List[int]:
        if len(self.bits) < n:
            nbytes = max(64, 2 * ((n + 7) // 8))
            raw = hashlib.shake_256(self.seed).digest(nbytes)
            self.bits = [(b >> (7 - i)) & 1 for b in raw for i in range(8)]
        return self.bits[:n]


class CapacityExceeded(ValueError):
    """The payload did not fit in ``max_new_tokens``.

    ``bits_done`` is how many payload bits the coder had committed when it hit the
    limit: a direct measurement of this passage's capacity, used by the sender to
    size its next attempt instead of searching blindly.
    """

    def __init__(self, message: str, bits_done: int, bits_total: int):
        super().__init__(message)
        self.bits_done = bits_done
        self.bits_total = bits_total


def payload_bit_length(nbytes: int, key: Optional[bytes]) -> int:
    """Length of the coded bitstream for ``nbytes`` of data (independent of content)."""
    return len(_payload_bits(bytes(nbytes), key, 0))


def _payload_bits(data: bytes, key: Optional[bytes], nonce: int) -> List[int]:
    frame = _frame_payload(data)
    if key is None:
        return frame
    ks = _Keystream(key, nonce).get(len(frame))
    return _int_bits(nonce, NONCE_BITS) + [a ^ b for a, b in zip(frame, ks)]


# --------------------------------------------------------------------------
# OD-Stega temperature
# --------------------------------------------------------------------------


def _kl(q: torch.Tensor, p: torch.Tensor) -> float:
    mask = q > 0
    return float((q[mask] * (q[mask].log() - p[mask].log())).sum())


def od_temperature(p_hat: torch.Tensor, budget: float, iters: int = 50) -> Tuple[torch.Tensor, float]:
    """Max-entropy Q with KL(Q || p_hat) <= budget: Q ∝ p_hat**beta, beta = 1/T.

    Returns (Q, T). Deterministic bisection in float64.
    """
    if budget <= 0:
        return p_hat, 1.0
    uniform = torch.full_like(p_hat, 1.0 / p_hat.numel())
    if _kl(uniform, p_hat) <= budget:
        return uniform, float("inf")
    logp = p_hat.log()
    lo, hi = 0.0, 1.0  # beta: KL is decreasing in beta; beta=1 -> KL=0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        q = torch.softmax(mid * logp, dim=0)
        if _kl(q, p_hat) > budget:
            lo = mid
        else:
            hi = mid
    q = torch.softmax(hi * logp, dim=0)
    return q, 1.0 / hi


_NL_CACHE: dict = {}


def _newline_sets(tokenizer):
    """(tokens containing "\n\n", tokens starting with "\n", tokens ending with "\n"), cached."""
    key = id(tokenizer)
    if key not in _NL_CACHE:
        ids = range(len(tokenizer))
        pieces = tokenizer.batch_decode([[i] for i in ids])
        _NL_CACHE[key] = (frozenset(i for i, t in zip(ids, pieces) if "\n\n" in t),
                          frozenset(i for i, t in zip(ids, pieces) if t.startswith("\n")),
                          frozenset(i for i, t in zip(ids, pieces) if t.endswith("\n")))
    return _NL_CACHE[key]


def _newline_banned(tokenizer, context_ids: Sequence[int]) -> frozenset:
    """Tokens that would put a blank line into the stego-text after ``context_ids``.

    Depends only on the context, so encoder and decoder ban the same tokens."""
    double, starts, ends = _newline_sets(tokenizer)
    return double | starts if context_ids and context_ids[-1] in ends else double


def show_text(text: str, cfg: "TextConfig") -> str:
    """What the sender publishes: line breaks doubled when ``newline_fix`` is on."""
    return text.replace("\n", "\n\n") if cfg.newline_fix else text


def unshow_text(text: str, cfg: "TextConfig") -> str:
    """Inverse of :func:`show_text`, applied by the receiver before decoding."""
    return text.replace("\n\n", "\n") if cfg.newline_fix else text


def step_distribution(logits: torch.Tensor, context_ids: Sequence[int], tokenizer, top_k: int, C: float,
                      guard: str = "filter", newline_fix: bool = False):
    """Alphabet, cumulative table, and diagnostics for one generation step."""
    scores = logits.detach().to(torch.float64)
    log_p = torch.log_softmax(scores, dim=0)
    p = log_p.exp()
    entropy = float(-(p * log_p).sum())  # nats

    ranked = torch.argsort(scores.to(torch.float32), descending=True, stable=True)[:top_k].tolist()
    if guard == "filter":
        alphabet = _stable_candidates(tokenizer, context_ids, ranked)
    else:
        specials = _special_ids(tokenizer)
        alphabet = [t for t in ranked if t not in specials]
    if newline_fix:
        banned = _newline_banned(tokenizer, context_ids)
        alphabet = [t for t in alphabet if t not in banned]
    if len(alphabet) < MIN_ALPHABET:
        raise ValueError(f"only {len(alphabet)} usable candidates at this step; change the prompt")

    idx = torch.tensor(alphabet, dtype=torch.long)
    p_sub = p[idx]
    mass = float(p_sub.sum())
    p_hat = p_sub / mass
    trunc_kl = -math.log(mass)  # KL(p_hat || p), exact

    budget = max(0.0, C * entropy - trunc_kl)
    q, temp = od_temperature(p_hat, budget)

    freqs = quantize_frequencies(q)
    cum = cumulative(freqs)
    q_used = torch.tensor(freqs, dtype=torch.float64) / FREQ_TOTAL
    kl_used = _kl(q_used, p_sub)  # vs. the full P, restricted to the support of q_used
    top = torch.tensor(ranked, dtype=torch.long)
    p_top = p[top] / p[top].sum()
    pos = {t: i for i, t in enumerate(ranked)}
    kl_cover = _kl(q_used, p_top[[pos[a] for a in alphabet]])
    return alphabet, cum, {"kl_nats": kl_used, "kl_cover_nats": kl_cover, "entropy_nats": entropy, "temperature": temp}


# --------------------------------------------------------------------------
# encode
# --------------------------------------------------------------------------


def _encode_once(bits: List[int], cfg: TextConfig, model, tokenizer):
    total = len(bits)
    prompt_ids = _token_ids(tokenizer, cfg.prompt)
    # Rolling context window (from subtext-codec): encoder and decoder roll at the
    # same steps, so messages longer than the model's context still decode exactly.
    stepper = _Stepper(model, cfg.device, prompt_ids, window=context_limit(model))
    ids = list(prompt_ids)
    reader = ArithmeticDecoder(bits + ENCODER_TAIL)
    mirror = ArithmeticEncoder()
    acc = {"kl": 0.0, "klc": 0.0, "ent": 0.0, "t": 0.0, "nt": 0}
    generated = stagnant = 0
    while len(mirror.bits) < total:
        if cfg.max_new_tokens is not None and generated >= cfg.max_new_tokens:
            raise CapacityExceeded(f"payload needs more than {cfg.max_new_tokens} tokens", len(mirror.bits), total)
        if stagnant >= _STALL_LIMIT:
            raise ValueError("arithmetic coder stopped making progress")
        before = len(mirror.bits)
        alphabet, cum, diag = step_distribution(stepper.logits, ids, tokenizer, cfg.top_k, cfg.C, cfg.guard, cfg.newline_fix)
        sym = reader.decode(cum)
        mirror.encode(sym, cum)
        token = alphabet[sym]
        ids.append(token)
        stepper.advance(token)
        generated += 1
        acc["kl"] += diag["kl_nats"]
        acc["klc"] += diag["kl_cover_nats"]
        acc["ent"] += diag["entropy_nats"]
        if math.isfinite(diag["temperature"]):
            acc["t"] += diag["temperature"]
            acc["nt"] += 1
        stagnant = 0 if len(mirror.bits) > before else stagnant + 1
    if mirror.bits[:total] != bits:
        raise ValueError("internal error: coder did not reproduce the payload")
    tail = _finish_sentence(stepper, ids, cfg, tokenizer) if cfg.finish_sentence else 0
    text = tokenizer.decode(ids, skip_special_tokens=True)
    round_trip = _token_ids(tokenizer, text)[: len(ids)] == ids and not (cfg.newline_fix and "\n\n" in text)
    if tail and not round_trip:  # the tail changed how the text re-reads: drop it
        del ids[len(ids) - tail:]
        tail = 0
        text = tokenizer.decode(ids, skip_special_tokens=True)
        round_trip = _token_ids(tokenizer, text)[: len(ids)] == ids and not (cfg.newline_fix and "\n\n" in text)
    acc["tail"] = tail
    return show_text(text, cfg), round_trip, generated, acc


_SENTENCE_END = (".", "!", "?", '."', '!"', '?"', ".'", ".)")


def _ends_sentence(tokenizer, token: int) -> bool:
    piece = tokenizer.decode([token])
    return "\n" in piece or piece.rstrip().endswith(_SENTENCE_END)


def _finish_sentence(stepper, ids: List[int], cfg: TextConfig, tokenizer) -> int:
    """Append ordinary top-k samples until a sentence ends; returns how many.

    Candidates go through the same stability filter as the payload tokens, so
    the appended words cannot merge with the last hidden token when the
    receiver re-tokenises the text.
    """
    if _ends_sentence(tokenizer, ids[-1]):
        return 0
    gen = torch.Generator().manual_seed(int.from_bytes(os.urandom(8), "big") >> 1)
    added = 0
    while added < cfg.max_tail_tokens:
        scores = stepper.logits.detach().to(torch.float64)
        ranked = torch.argsort(scores.to(torch.float32), descending=True, stable=True)[: cfg.top_k].tolist()
        cands = _stable_candidates(tokenizer, ids, ranked)
        if cfg.newline_fix:
            banned = _newline_banned(tokenizer, ids)
            cands = [t for t in cands if t not in banned]
        if not cands:
            break
        probs = torch.softmax(scores[torch.tensor(cands)], dim=0)
        token = cands[int(torch.multinomial(probs, 1, generator=gen))]
        ids.append(token)
        stepper.advance(token)
        added += 1
        if _ends_sentence(tokenizer, token):
            break
    return added


def encode(data: bytes, cfg: TextConfig, model, tokenizer, key: Optional[bytes] = None,
           nonce: Optional[int] = None) -> Tuple[str, TextStats]:
    """Hide ``data`` in text generated from ``cfg.prompt``.

    With ``key``, the bitstream is nonce + whitened frame (recommended). ``nonce``
    fixes the first nonce for reproducible tests; by default it is random.
    """
    if cfg.guard not in ("filter", "retry"):
        raise ValueError("guard must be 'filter' or 'retry'")
    if cfg.guard == "retry" and key is None:
        raise ValueError("the retry guard needs a key (a fresh nonce is what changes between attempts)")
    n0 = int.from_bytes(os.urandom(4), "big") if nonce is None else nonce
    tries = MAX_RETRIES if cfg.guard == "retry" else 1
    for attempt in range(tries):
        bits = _payload_bits(data, key, (n0 + attempt) & 0xFFFFFFFF)
        text, ok, generated, acc = _encode_once(bits, cfg, model, tokenizer)
        if ok:
            break
    else:
        raise ValueError(f"no attempt out of {tries} re-tokenised cleanly; use a shorter message or the filter guard")
    if not ok:
        raise ValueError("generated text does not tokenize back to the same tokens")
    n = max(generated, 1)
    stats = TextStats(
        tokens=generated,
        payload_bytes=len(data),
        bits_per_token=8 * len(data) / n,
        kl_bits_total=acc["kl"] / LN2,
        kl_bits_per_token=acc["kl"] / LN2 / n,
        kl_vs_topk_bits_per_token=acc["klc"] / LN2 / n,
        mean_entropy_bits=acc["ent"] / LN2 / n,
        mean_temperature=(acc["t"] / acc["nt"]) if acc["nt"] else float("inf"),
        attempts=attempt + 1,
        tail_tokens=acc.get("tail", 0),
    )
    return text, stats


# --------------------------------------------------------------------------
# decode
# --------------------------------------------------------------------------


def _body_ids(text: str, cfg: TextConfig, tokenizer):
    start = text.find(cfg.prompt)
    if start == -1:
        raise ValueError("prompt not found in stego-text")
    prompt_ids = _token_ids(tokenizer, cfg.prompt)
    ids = _token_ids(tokenizer, text[start:])
    if ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("stego-text does not tokenize to prompt + message")
    return prompt_ids, ids[len(prompt_ids):]


def decode(text: str, cfg: TextConfig, model, tokenizer, key: Optional[bytes] = None) -> bytes:
    data, _ = decode_with_stats(text, cfg, model, tokenizer, key)
    return data


def decode_with_stats(text: str, cfg: TextConfig, model, tokenizer, key: Optional[bytes] = None):
    """Decode; also return per-token KL diagnostics (vs full P and vs top-k)."""
    text = unshow_text(text, cfg)
    prompt_ids, body = _body_ids(text, cfg, tokenizer)
    stepper = _Stepper(model, cfg.device, prompt_ids, window=context_limit(model))
    context = list(prompt_ids)
    writer = ArithmeticEncoder()
    offset = NONCE_BITS if key is not None else 0
    ks: Optional[_Keystream] = None
    needed: Optional[int] = None
    kl = klc = 0.0
    n = 0

    def frame_bits(upto: Optional[int] = None) -> List[int]:
        raw = writer.bits[offset:] if upto is None else writer.bits[offset:offset + upto]
        if ks is None:
            return list(raw)
        return [a ^ b for a, b in zip(raw, ks.get(len(raw)))]

    for token in body:
        if needed is not None and len(writer.bits) >= offset + needed:
            break
        alphabet, cum, diag = step_distribution(stepper.logits, context, tokenizer, cfg.top_k, cfg.C, cfg.guard, cfg.newline_fix)
        if token not in alphabet:
            break
        writer.encode(alphabet.index(token), cum)
        kl += diag["kl_nats"]
        klc += diag["kl_cover_nats"]
        n += 1
        context.append(token)
        stepper.advance(token)
        if key is not None and ks is None and len(writer.bits) >= NONCE_BITS:
            ks = _Keystream(key, _bits_int(writer.bits[:NONCE_BITS]))
        if needed is None and (key is None or ks is not None):
            needed = _declared_length(frame_bits())
    n = max(n, 1)
    data = _unframe_payload(frame_bits(needed))
    return data, {"tokens": n, "kl_bits_per_token": kl / LN2 / n, "kl_vs_topk_bits_per_token": klc / LN2 / n}


# --------------------------------------------------------------------------
# cover (non-stego) text, for detector experiments
# --------------------------------------------------------------------------


def sample_cover(cfg: TextConfig, model, tokenizer, n_tokens: int, seed: int) -> str:
    """Ordinary top-k sampling at temperature 1 — what an innocent user would post.

    No guard and no coder: this is the reference "normal" distribution a warden
    compares stego-text against.
    """
    gen = torch.Generator().manual_seed(seed)
    prompt_ids = _token_ids(tokenizer, cfg.prompt)
    stepper = _Stepper(model, cfg.device, prompt_ids, window=context_limit(model))
    ids = list(prompt_ids)
    for _ in range(n_tokens):
        scores = stepper.logits.detach().to(torch.float64)
        top = torch.topk(scores, cfg.top_k)
        probs = torch.softmax(top.values, dim=0)
        choice = int(top.indices[torch.multinomial(probs, 1, generator=gen)])
        ids.append(choice)
        stepper.advance(choice)
    return tokenizer.decode(ids, skip_special_tokens=True)
