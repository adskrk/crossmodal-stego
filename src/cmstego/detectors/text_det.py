"""Text steganalysis used as the warden's "text eye".

Both detectors are trained to separate *stego-text* from *cover text*, where
cover text is ordinary top-k sampling from the same language model (see
``text_channel.sample_cover``). That is the honest comparison for generative
steganography: the question is whether hiding a message changes the model's
output distribution, not whether a small model writes like a human.

* ``NgramDetector`` — TF-IDF over word and character n-grams + logistic
  regression. A black-box warden: sees only the text.
* ``LikelihoodDetector`` — a white-box warden that has the same language
  model. For each text it computes statistics of the per-token log-probability
  under the model (mean, std, quantiles, fraction of tokens outside the top-k)
  and feeds them to logistic regression. Raising the OD budget C flattens the
  distribution, so stego tokens become less probable — this detector is
  designed to catch exactly that.
* ``StyleDetector`` — a black-box warden of a different family: hand-made
  writing-style statistics (word and sentence lengths, vocabulary richness,
  punctuation, repetition) + logistic regression. Needs no language model, so
  it can be computed from saved texts. Used as a third text eye so that a
  selection policy can be trained against some detectors and tested on others.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler


class NgramDetector:
    def __init__(self):
        feats = FeatureUnion([
            ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)),
        ])
        self.pipe = make_pipeline(feats, LogisticRegression(max_iter=2000, C=1.0))

    def fit(self, texts: Sequence[str], labels: Sequence[int]):
        self.pipe.fit(list(texts), list(labels))
        return self

    def score(self, texts: Sequence[str]) -> np.ndarray:
        return self.pipe.predict_proba(list(texts))[:, 1]


def likelihood_features(texts: Sequence[str], prompt: str, model, tokenizer, top_k: int = 64) -> np.ndarray:
    out = []
    p_ids = tokenizer(prompt)["input_ids"]
    limit = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", 1024)
    for t in texts:
        ids = tokenizer(t)["input_ids"][:limit]  # score at most one context window
        x = torch.tensor([ids])
        with torch.no_grad():
            logits = model(input_ids=x).logits[0].to(torch.float64)
        logp = torch.log_softmax(logits[:-1], dim=-1)
        tgt = x[0, 1:]
        start = max(len(p_ids) - 1, 0)
        lp = logp[torch.arange(len(tgt)), tgt][start:].numpy()
        ranks = (logp[start:] > logp[torch.arange(len(tgt)), tgt][start:, None]).sum(-1).numpy()
        ent = -(logp[start:].exp() * logp[start:]).sum(-1).numpy()
        if len(lp) == 0:
            lp, ranks, ent = np.zeros(1), np.zeros(1), np.zeros(1)
        out.append([lp.mean(), lp.std(), np.quantile(lp, 0.1), np.quantile(lp, 0.5),
                    (ranks >= top_k).mean(), (ranks == 0).mean(), (-lp - ent).mean()])
    return np.asarray(out)


class LikelihoodDetector:
    def __init__(self, prompt_of, model, tokenizer, top_k: int = 64):
        self.prompt_of = prompt_of  # function: text -> prompt it was generated from
        self.model, self.tok, self.k = model, tokenizer, top_k
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    def _feats(self, texts):
        rows = [likelihood_features([t], self.prompt_of(t), self.model, self.tok, self.k)[0] for t in texts]
        return np.asarray(rows)

    def fit(self, texts, labels):
        self.clf.fit(self._feats(texts), list(labels))
        return self

    def score(self, texts) -> np.ndarray:
        return self.clf.predict_proba(self._feats(texts))[:, 1]


_STOP = frozenset("the a an and or of to in on for with is are was were be been it this that as at by from "
                  "you we they he she i not but if so can will".split())


def style_features(text: str) -> List[float]:
    import re

    words = re.findall(r"[A-Za-z']+", text)
    n = max(len(words), 1)
    lens = np.array([len(w) for w in words]) if words else np.zeros(1)
    lower = [w.lower() for w in words]
    uniq = set(lower)
    counts = {}
    for w in lower:
        counts[w] = counts.get(w, 0) + 1
    hapax = sum(1 for c in counts.values() if c == 1)
    sents = [x for x in re.split(r"[.!?]+", text) if x.strip()]
    slens = np.array([len(re.findall(r"[A-Za-z']+", x)) for x in sents]) if sents else np.zeros(1)
    bigrams = list(zip(lower, lower[1:]))
    rep_bigrams = 1 - len(set(bigrams)) / max(len(bigrams), 1)
    chars = max(len(text), 1)
    return [
        np.log(n), lens.mean(), lens.std(), len(uniq) / n, hapax / n,
        sum(ch in ",;:" for ch in text) / chars, sum(ch in ".!?" for ch in text) / chars,
        text.count("\n") / chars, sum(ch.isdigit() for ch in text) / chars,
        sum(w[0].isupper() for w in words) / n, slens.mean(), slens.std(),
        sum(ord(ch) > 127 for ch in text) / chars, sum(ch in "\"'()" for ch in text) / chars,
        sum(w in _STOP for w in lower) / n, rep_bigrams,
    ]


class StyleDetector:
    def __init__(self):
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.5))

    def fit(self, texts: Sequence[str], labels: Sequence[int]):
        self.pipe.fit(np.array([style_features(t) for t in texts]), list(labels))
        return self

    def score(self, texts: Sequence[str]) -> np.ndarray:
        return self.pipe.predict_proba(np.array([style_features(t) for t in texts]))[:, 1]
