"""Build a tiny offline GPT-2-style model for tests when Hugging Face is unreachable.

The text channel is model-agnostic; real experiments use `gpt2` or SmolLM2.
This toy model exists so the codec can be tested end to end without a network:
a byte-level BPE tokenizer (same scheme as GPT-2) and a 4-layer GPT-2 trained
for a few minutes on English documentation found on this machine.

Usage:  python scripts/make_toy_lm.py --out data/toy-lm --minutes 10
"""

import argparse
import glob
import os
import random
import sys
import time

import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


def collect_corpus(max_chars: int) -> str:
    roots = [p for p in sys.path if p.endswith("site-packages")]
    files = []
    for r in roots:
        for ext in ("md", "rst", "txt"):
            files += glob.glob(os.path.join(r, "**", f"*.{ext}"), recursive=True)
    random.Random(0).shuffle(files)
    chunks, total = [], 0
    for f in files:
        if any(s in f.lower() for s in ("license", "record", "entry_points", "top_level", "requires", "vocab", "merges")):
            continue
        try:
            t = open(f, encoding="utf-8").read()
        except Exception:
            continue
        letters = sum(c.isalpha() for c in t)
        if len(t) < 500 or letters / max(len(t), 1) < 0.6:
            continue  # skip code-heavy or tiny files
        chunks.append(t)
        total += len(t)
        if total >= max_chars:
            break
    # Docstrings from Python's standard library: plenty of plain English prose.
    import ast
    stdlib = os.path.dirname(os.__file__)
    for f in sorted(glob.glob(os.path.join(stdlib, "**", "*.py"), recursive=True)):
        if total >= max_chars or "test" in f:
            continue
        try:
            tree = ast.parse(open(f, encoding="utf-8").read())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                d = ast.get_docstring(node)
                if d and len(d) > 200:
                    chunks.append(d)
                    total += len(d)
    return "\n\n".join(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/toy-lm")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--vocab", type=int, default=4096)
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(args.out, exist_ok=True)

    text = collect_corpus(6_000_000)
    print(f"corpus: {len(text):,} chars")
    corpus_path = os.path.join(args.out, "corpus.txt")
    open(corpus_path, "w", encoding="utf-8").write(text)

    bpe = ByteLevelBPETokenizer()
    bpe.train([corpus_path], vocab_size=args.vocab, min_frequency=2, special_tokens=["<|endoftext|>"])
    bpe.save_model(args.out)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=bpe._tokenizer,
        eos_token="<|endoftext|>", bos_token="<|endoftext|>", unk_token="<|endoftext|>",
    )
    tok.model_max_length = 10**9  # only for tokenising the corpus; the model caps context
    ids = torch.tensor(bpe.encode(text).ids, dtype=torch.long)
    assert len(ids) > 10_000, "corpus tokenised to almost nothing"
    print(f"tokens: {len(ids):,}")

    cfg = GPT2Config(vocab_size=len(tok), n_positions=256, n_embd=192, n_layer=4, n_head=4,
                     bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    model = GPT2LMHeadModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    block, batch = 128, 16
    t0, step = time.time(), 0
    model.train()
    while time.time() - t0 < args.minutes * 60:
        starts = torch.randint(0, len(ids) - block - 1, (batch,))
        x = torch.stack([ids[s : s + block] for s in starts])
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    os.remove(corpus_path)
    print(f"saved to {args.out} after {step} steps, final loss {loss.item():.3f}")


if __name__ == "__main__":
    main()
