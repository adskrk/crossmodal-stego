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
