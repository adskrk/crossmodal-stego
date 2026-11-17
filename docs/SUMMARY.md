# Summary

**Project:** Reinforcement learning and LLMs for cross-modal steganography.
One secret is sealed into an all-or-nothing package and split between a
GPT-2-written paragraph and an RGB photo; a pooled warden inspects both; the
sender chooses carriers to stay unnoticed.

All experiment numbers below come from `docs/RESULTS.md`, which
`scripts/make_report.py` builds from saved outputs. Small scale: 195 photo
crops, 12 prompts, GPT-2 (124M) on a laptop CPU.

## What works (built and tested)

| Part | Status | Evidence |
| --- | --- | --- |
| Lock-box package: secret compressed, locked with a fresh key, key mixed in (Rivest's package transform) | done | either carrier's part alone opens nothing; unit tests |
| Framing: split, headers, CRC, whitening | done | unit tests |
| Text channel: arithmetic coding + OD-Stega budget + exact KL, line-break fix, natural sentence ending | done | 864 / 864 GPT-2 stego-texts recovered exactly across all runs (1078 / 1078 with the toy model) |
| Image channel: LSB replacement, LSB matching, HILL + STC on RGB | done | 2135 / 2135 stego images recovered exactly |
| Sender / receiver, browser app, damage reporting | done | 87 automated tests; JPEG damage reported as "image fragment missing" |
| Detectors: SPA, 3 pretrained CNNs, SPAM-LR, n-gram, likelihood, style | done | AUC, P_E and detection at 5% false alarms, with 95% intervals |
| Pooled warden (p-value fusion, 5% false alarms) | done | used in the selection experiments |
| Carrier selection: baselines, grid search, PPO; one warden and an ensemble | done | held-out evaluation, unseen warden |
| Runs natively on Windows (Python 3.11) | done | 87 / 87 tests on the author's Windows laptop |

## Findings (measured, not assumed)

1. **A leak in the reused text coder, found and fixed.** subtext-codec sends its
   length field unscrambled. On GPT-2 an n-gram detector caught it with AUC
   0.99; a random nonce plus whitening of the whole bitstream brought it to
   0.55 [0.44, 0.66], indistinguishable from chance, at the same capacity
   (2.42 vs 2.39 bits/token).
2. **The OD-Stega budget trades capacity for detectability.** On GPT-2, C = 0.1
   carries 29% more (3.08 vs 2.39 bits/token) but the white-box detector
   catches every stego-text (AUC 1.00). The system uses C = 0.
3. **A structural fingerprint of exact-decoding text steganography, found and
   largely fixed.** A detector that knows the model reached AUC 0.76 at C = 0 on
   GPT-2. Its visible form: **71% of ordinary GPT-2 texts contain a blank line, but
   only 4% of stego-texts.** GPT-2 writes a blank line as one token, and the word
   after it re-reads as two line breaks plus the word, so exact decoding cannot
   allow it. A style detector with no language model sees this (AUC 0.67). **The
   fix:** ban blank-line tokens, show every line break doubled, let the receiver
   halve them back. Final GPT-2 run (192 messages, 96 test pairs): blank lines match
   (75% vs 71%), style detector 0.50 (chance), **white-box detector 0.60
   [0.52, 0.69]**, catching 3% of stego-texts at a 5% false-alarm rate (chance
   level), for less than 1% less capacity (2.38 vs 2.39 bits/token). Not perfect:
   the white-box and n-gram detectors stay just above 0.5 in AUC. A first, smaller
   run (48 pairs) had suggested 0.52; the larger run is the one to quote. The fix
   is the default.
4. **A capacity bug in HILL + STC on real photos, found and fixed.** Photos with
   blown-out or flat regions broke pySTC silently; a per-photo capacity rule plus
   verify-after-embed now refuses such payloads (205 refusals, 0 broken images).
5. **Image detectability depends on the photo source.** SPA catches LSB
   replacement (AUC up to 0.99) and nothing else. Pretrained CNNs are at chance on
   Kodak but detect HILL on scikit-image photos (AUC 0.61 / 0.70 / 0.81 at 0.05 /
   0.1 / 0.2 bits per value; P_E 0.37 / 0.31 / 0.21).
6. **Search overfits; one learned policy did not.** First runs (no line-break
   fix, 4 test texts per prompt): every searched or learned strategy reached about
   0% flags under the warden(s) it was tuned on, but none clearly beat *no
   selection* (21%) under the unseen warden B; grid search reached 49%. **Final run**
   (line-break fix on, 16 texts per prompt):
   - PPO with the scoring policy, trained only against warden A: A's flag rate
     0.7% vs 15.5% for no selection; unseen warden B 2.8% vs 4.1% on unseen Kodak
     photos, and 0.2% vs 7.6% on the unseen photo source; also lower than no
     selection under the warden ensemble (A, C, D combined: 12% vs 22% and 18% vs
     29%), whose C and D parts it never saw; C and D were not evaluated
     separately. **The first learned policy here that does not overfit its
     training detector** (one seed so far);
   - grid search still overfits: 0% under A, 35% under B, mostly by choosing C = 0.1;
   - PPO trained against the ensemble learned poorly (32% under A);
   - the heuristic is perfect under B but the worst under A.
7. **Choosing the split did not help.** With the lock-box, the text share can be
   chosen for detectability alone. Under the unseen warden B the smallest share
   (24-byte fragment, ~81 tokens) had the lowest mean suspicion, 0.69, against
   0.83–0.88 for every strategy (grid search, PPO) that picked longer texts; longer
   texts are slightly more detectable (white-box AUC 0.76 → 0.81–0.82). The
   experiment's flag rates are unusable because 24 calibration texts cannot give a
   p-value below the threshold. The supported rule is simply: keep the text short.
8. **Speed.** Sizing the text from the coder's measured capacity instead of a
   bisection cut sending time about 5× on GPT-2 (1,000-byte secret: 93 s → 16 s;
   one line: 51 s → 10 s), measured with `scripts/bench_send.py` on the laptop.

## What is new here

- Splitting a secret across carriers is **not** new (batch steganography,
  multimodal allocation papers), and the package transform is Rivest's.
- What this implementation adds: a **text + image** system whose split is
  all-or-nothing, judged by a **pooled warden with independent detector sets**;
  a **simple, measured fix for the blank-line form of the tokenisation
  fingerprint** that brings a white-box GPT-2 detector from AUC 0.76 to 0.60 at
  C = 0 (chance-level detection at a 5% false-alarm rate);
  and measurements of failure modes: the clear length field, and proxy-detector
  overfitting of selection policies. Whether these are new to the literature still
  needs a proper search before any claim.
- The RL contribution: search-based selection overfits the detectors it is tuned
  on, but a PPO scoring policy trained against one warden stayed below no
  selection under the fully unseen warden B and under a warden ensemble it was not
  trained on, in the final run (one seed, to be repeated). A learned split did no
  better than "keep the text short".

## Limitations

- Few texts per prompt in the first selection runs (4 held-out per prompt and
  budget; 8 in the final run) and at the longer split sizes (2), where the flag
  rates are unusable.
- The final PPO result is from one training seed, and wardens C and D were only
  evaluated as part of the ensemble.
- The line-break fix lowers but does not remove detectability: the white-box and
  n-gram detectors stay at AUC 0.60, slightly above chance. A weaker trace remains:
  in the first 48-pair run, 23 of 96 fixed stego-texts (vs 5 of 96 ordinary texts)
  contained a token outside the model's top 64 when re-read.
- The split experiment used stego-texts without the line-break fix.
- 256×256 crops from two small photo sources; pretrained CNNs were trained on
  512×512 ALASKA2 photos, so they are mismatched by design.
- Selection uses cached measurements with linear interpolation between payload
  rates; final claims need fresh end-to-end encodes.
