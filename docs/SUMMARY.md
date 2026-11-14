# Summary

**Project:** Reinforcement learning and LLMs for cross-modal steganography.
One secret is sealed into an all-or-nothing package and split between a
GPT-2-written paragraph and an RGB photo; a pooled warden inspects both; the
sender chooses carriers (and, next, the split) to stay unnoticed.

All experiment numbers below come from `docs/RESULTS.md`, which
`scripts/make_report.py` builds from saved outputs. Small scale: 195 photo
crops, 12 prompts, GPT-2 (124M) on a laptop CPU.

## What works (built and tested)

| Part | Status | Evidence |
| --- | --- | --- |
| Lock-box package: secret compressed, locked with a fresh key, key mixed in (Rivest's package transform) | done | either carrier's part alone opens nothing; unit tests |
| Framing: split, headers, CRC, whitening | done | unit tests |
| Text channel: arithmetic coding + OD-Stega budget + exact KL, natural sentence ending | done | 288 / 288 GPT-2 stego-texts recovered exactly (1078 / 1078 with the toy model) |
| Image channel: LSB replacement, LSB matching, HILL + STC on RGB | done | 2135 / 2135 stego images recovered exactly |
| Sender / receiver, browser app, damage reporting | done | 87 automated tests; JPEG damage reported as "image fragment missing" |
| Detectors: SPA, 3 pretrained CNNs, SPAM-LR, n-gram, likelihood, style | done | AUC, P_E and detection at 5% false alarms, with 95% intervals |
| Pooled warden (p-value fusion, 5% false alarms) | done | used in the selection experiments |
| Carrier selection: baselines, grid search, PPO; one warden and an ensemble | done | held-out evaluation, unseen warden |
| Runs natively on Windows (Python 3.11) | done | 76 / 76 tests on the author's laptop before this update |

## Findings (measured, not assumed)

1. **A leak in the reused text coder, found and fixed.** subtext-codec sends its
   length field unscrambled. On GPT-2 an n-gram detector caught it with AUC
   0.99; a random nonce plus whitening of the whole bitstream brought it to
   0.55 [0.44, 0.66], indistinguishable from chance, at the same capacity
   (2.42 vs 2.39 bits/token).
2. **The OD-Stega budget trades capacity for detectability.** On GPT-2, C = 0.1
   carries 29% more (3.08 vs 2.39 bits/token) but the white-box detector
   catches every stego-text (AUC 1.00). The system uses C = 0.
3. **A structural fingerprint of exact-decoding text steganography.** A detector
   that knows the model still reaches AUC 0.76 at C = 0 on GPT-2 (0.71 on the
   toy model). Its visible form: **71% of ordinary GPT-2 texts contain a blank
   line, but only 4% of stego-texts**. GPT-2 writes a blank line as one token,
   and the word after it re-reads as two line breaks plus the word; exact
   decoding cannot allow that, so stego-text uses single line breaks. A simple
   style detector with no language model sees this (AUC 0.67 [0.57, 0.78]); the
   n-gram detector, which ignores line breaks, does not. **A fix is built**
   (blank-line tokens banned, line breaks shown doubled and halved again by the
   receiver; exact on the toy model) and its GPT-2 measurement is in the
   overnight run.
4. **A capacity bug in HILL + STC on real photos, found and fixed.** Photos with
   blown-out or flat regions broke pySTC silently; a per-photo capacity rule plus
   verify-after-embed now refuses such payloads (205 refusals, 0 broken images).
5. **Image detectability depends on the photo source.** SPA catches LSB
   replacement (AUC up to 0.99) and nothing else. Pretrained CNNs are at chance on
   Kodak but detect HILL on scikit-image photos (AUC 0.61 / 0.70 / 0.81 at 0.05 /
   0.1 / 0.2 bits per value; P_E 0.37 / 0.31 / 0.21).
6. **Optimising against known detectors overfits them, even several at once.**
   With GPT-2 text, every search or learned strategy reached about 0% flags under
   the warden(s) it was tuned on, but none clearly beat *no selection* (21%)
   under the unseen warden B:
   - grid search vs warden A: 49%; vs an ensemble of three wardens sharing no
     detector with B: 47%. Both pick C = 0.1 for about 40% of messages, which
     black-box detectors partly miss and B's white-box detector always catches;
   - PPO vs the ensemble: 16% [13%, 18%] on Kodak, the lowest, and 0% on the
     second photo source. That 0% comes from always choosing one test prompt
     whose 4 held-out texts B happens not to flag, so it is weak evidence;
   - the heuristic (busiest photo, highest-entropy prompt, C = 0): 24%.
7. **Speed.** Sizing the text from the coder's measured capacity instead of a
   bisection cut sending time about 5× on GPT-2 (1,000-byte secret: 93 s → 16 s;
   one line: 51 s → 10 s), measured with `scripts/bench_send.py` on the laptop.

## What is new here, honestly stated

- Splitting a secret across carriers is **not** new (batch steganography,
  multimodal allocation papers), and the package transform is Rivest's (1997).
- What this implementation adds: a **text + image** system whose split is
  all-or-nothing, judged by a **pooled warden with independent detector sets**,
  and measurements of concrete failure modes: the clear length field, the
  tokenisation fingerprint and its blank-line form, and proxy-detector
  overfitting of selection policies. Whether these are new to the literature
  still needs a proper search before any claim.
- The RL contribution is **open**: selection matters, but every strategy tuned
  on known detectors so far overfits them. Letting RL choose the split is built
  and waits for its GPT-2 measurements.

## Limitations

- Few texts per prompt (4 held-out per prompt and budget in the selection
  experiments); single lucky prompts can dominate a policy's score.
- 256×256 crops from two small photo sources; pretrained CNNs were trained on
  512×512 ALASKA2 photos, so they are mismatched by design.
- Selection uses cached measurements with linear interpolation between payload
  rates; final claims need fresh end-to-end encodes.
- No robustness to editing or recompression: both carriers are exact-bit schemes.
- The lock-box construction has not had an independent security review.
