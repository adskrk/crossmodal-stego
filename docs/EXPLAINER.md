# How the system works — plain-language guide

---

## 1. The big picture

One secret message is cut into two pieces. One piece hides inside a paragraph
written by a language model; the other hides inside a colour photo. Neither
carrier looks like it is hiding anything. The receiver, who shares a key and
the same model, pulls both pieces out and checks that they join back into the
exact original message.

Then we ask the research question: **a warden who looks at the text and the
photo together — how suspicious does the pair look, and can the sender lower
that by choosing *which* photo, *which* prompt and *how aggressive* the text
coding is?**

```
secret ──► framing ──► split ──► text fragment ──► LLM arithmetic coder ──► stego-text ─┐
                              └► image fragment ─► HILL + STC embedding ──► stego-image ┤
                                                                                        ▼
                                                     pooled warden (text eye + image eye)
                                                                                        │
receiver: decode text, extract image, check CRCs, join ──► exact secret  ◄──────────────┘
```

---

## 2. Framing (`src/cmstego/framing.py`)

**What it does.** Splits the secret into a text fragment and an image fragment.
Each fragment starts with a 10-byte header: message id, where this piece starts
(offset), total length, and a CRC32 checksum of the whole message.

**Why.** If either carrier is damaged, the receiver knows *which one* failed,
instead of silently returning garbage. The message-level CRC also catches
mixing up pieces from two different messages.

**Whitening.** Each fragment is XORed with a keystream made from the shared key
(SHAKE-256). Arithmetic-coding steganography assumes the hidden bits look
random; plain English text does not. Whitening makes them random-looking. We
do *not* claim this is encryption-grade security; it is an ablation switch.

---

## 3. Text channel (`src/cmstego/text_channel.py`)

**Idea: arithmetic coding in reverse.** A language model gives, at every step,
a probability for each possible next word-piece (token). Arithmetic coding
normally *compresses* text using those probabilities. We run it backwards: the
secret bits act as the "random numbers" that pick each next token. A token the
model gives probability p carries about −log₂ p bits of the secret.

**Receiver.** Runs the same model on the same prompt, sees the same
probabilities, and runs the arithmetic *encoder* over the received tokens,
which gives back the secret bits exactly.

**Why the output looks normal.** With C = 0, each token is chosen with
(almost exactly) the model's own probability, so the stego-text is
statistically like ordinary samples from that model.

**What we reused.** subtext-codec (MIT licence): its arithmetic coder, its
length + CRC framing, its KV-cached stepping, and its *tokenisation guard*
(it only allows tokens that, when the text is written out and re-read, come
back as the same token — otherwise the receiver would see different tokens).

**What we added.**
1. **OD-Stega budget C** (EACL 2026). At each step we allow the distribution to
   change by a small KL amount δ = C × H(P) and pick the change that adds the
   most capacity. The paper proves that change is a temperature increase; we
   find the temperature by bisection. C = 0 → no change. Bigger C → more bits
   per token but easier to detect.
2. **Exact KL accounting.** For every token we compute the KL divergence
   between the integer table the coder *actually used* and the model's full
   distribution. That includes truncation to the top-64 tokens, the
   tokenisation guard, the temperature and rounding — nothing is left out.

3. **Nonce + whitening (found by experiment).** subtext-codec puts the message
   length at the very start of the bitstream, unscrambled. Equal-length messages
   therefore start with the same bits — and the same opening words. Our first
   detector run caught this at AUC 0.96. Now the bitstream starts with 32
   random bits (the nonce), and everything after it is XORed with a keystream
   made from the shared key and the nonce. The receiver reads the nonce first,
   rebuilds the keystream, and unscrambles. Result: AUC 0.52 (chance) for the
   text-only detector.
4. **Two tokenisation guards.** The receiver re-tokenises the text, so the
   sender's tokens must survive being written out and read back.
   - *filter* (subtext-codec, our default): only offer tokens that survive.
   - *retry* (OD-Stega): offer all top-k tokens; if the finished text does not
     re-tokenise cleanly, try again with a new nonce.
   Both produce text that always re-tokenises cleanly. Ordinary samples don't
   (40% contain a merge somewhere), and a detector holding the model can see
   that (AUC 0.71). So retry did not help and cost twice as many encodes.

---

## 4. Image channel (`src/cmstego/image_channel.py`)

The photo is treated as one long list of H × W × 3 colour values.

| Method | How it hides a bit | Weakness |
| --- | --- | --- |
| LSB replacement | overwrite the last bit of a value | only ever swaps 2k ↔ 2k+1 — a fingerprint that SPA/RS attacks read |
| LSB matching | if the last bit is wrong, add or subtract 1 at random | same amount of noise, no swap fingerprint |
| HILL + STC | each ±1 change has a *cost* (cheap in textured areas, expensive in flat areas); syndrome-trellis codes find a change pattern that carries the message at near-minimum total cost | strongest; still detectable at high rates |

**Why PSNR is high for all of them.** Each method changes values by at most 1.
For a random message at α bits per value, about half the touched values
change, so MSE ≈ α/2 and PSNR ≈ 10·log₁₀(255² / (α/2)), which is above 50 dB.
A test checks this formula. So PSNR does not tell the methods apart; detectors do.

**Capacity rule (found by experiment).** Photos with blown-out sky or black
borders have many values that cannot change (stuck at 0/255, or in perfectly
flat areas where HILL's cost is effectively infinite). pySTC failed silently on
them. We now count only *usable* values, cap HILL + STC at 0.4 bits per usable
value, refuse larger payloads, and decode every stego image once before
returning it. We also replaced pySTC's fixed 64-value length header with a
1024-value one, because 64 values can all land in a flat area.

---

## 5. Detectors and the pooled warden

**Image eye.**
- *SPA* (Sample Pairs Analysis): a formula, no training. Estimates how much
  LSB-replacement payload an image holds. Should be near zero for LSB
  matching and HILL.
- *Aletheia EfficientNet-B0* (pretrained, MIT): deep networks trained by
  Aletheia's author on ALASKA2 photos at 512 × 512. We use them unchanged on our
  256 × 256 crops of different photos — a deliberate cover-source mismatch.
- *SPAM + logistic regression*: our own small detector trained on our covers
  (out-of-fold, so it is never scored on photos it trained on).

**Text eye.**
- *N-gram detector*: TF-IDF on words and character groups + logistic regression.
- *Likelihood detector*: knows the language model; looks at how probable each
  token was. Built to catch a raised C.

Both are trained to tell stego-text from **ordinary samples of the same
model**, cut to the same length. That is the right comparison for generative
steganography: not "does it look human?" but "does hiding a message change
what the model would normally write?"

**Pooling.** Detector scores are not comparable to each other, so each is turned
into a p-value against clean carriers ("how unusual is this score among
innocent carriers?"). The warden flags a pair if either p-value is below 2.5%
(so about 5% false alarms on innocent pairs). Suspicion = 1 − min(p_text, p_image).

---

## 6. Carrier selection and RL (`src/cmstego/selection/carrier.py`)

**The decision.** For each message: pick 1 of 8 candidate photos, 1 of 4
candidate prompts, and C from {0, 0.02, 0.05, 0.1}.

**Why RL here and not for the split.** If the text coder at C = 0 is close to
undetectable, the best split is simple: fill a natural-length text, put the
rest in the image. What actually changes detectability is *which* carriers you
use. RL learns that choice from cheap features (image texture, brightness,
prompt entropy) without embedding into every candidate.

**How it is trained without days of compute.** Every candidate's detector score
was measured once, on real embeddings, and cached. The RL environment looks
scores up. The only approximation is linear interpolation of an image's score
between measured payload rates.

**Two wardens.** PPO is trained against warden A and evaluated against warden B
too, which it never saw. A policy that only exploits one detector's quirks
would do well on A and badly on B.

**Baselines.** No selection; random; a hand-written heuristic (busiest image,
highest-entropy prompt); grid search (embed into all 128 combinations and keep
the best — strong but expensive).

**Two PPO designs.**
- *v1 (slot index):* the action is "pick candidate 3". A plain neural network
  finds it hard to map 8 feature slots to the right index; it barely learned.
- *v2 (scoring):* the action is a weight vector; every candidate gets
  score = weights · features and the best one is chosen. Easier to learn and
  works for any number of candidates. It learned — but it learned to exploit
  warden A: C barely changed A's reward, so it always chose C = 0.1, which
  warden B catches every time.

**Possible question:** *Is this a real RL problem?* — It is a contextual bandit
(one decision per episode), the simplest RL setting. The multi-step version —
a warden that collects evidence over many messages — is the planned extension.


---

## 7. What is honest to claim from the results

See `docs/SUMMARY.md` (one page) and `docs/RESULTS.md` (all tables).
Short version: the system works end to end with exact recovery; the experiments
found and fixed two real flaws, measured a tokenisation fingerprint, and
showed carrier choice matters while RL is not yet better than a heuristic.
Claims of "better" need the larger runs planned after approval.

## 8. How AI was used

Claude (an AI assistant) wrote the code, tests and experiment scripts and ran
them. Every number was produced by running those scripts; none was typed in
by hand. You should be able to explain each module from this guide, rerun
every experiment, and check every number in RESULTS.md against `results/`.
