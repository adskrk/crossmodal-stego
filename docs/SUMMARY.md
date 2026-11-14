# Summary

**Project:** Reinforcement learning and LLMs for cross-modal steganography.
One secret is split between an LLM-written paragraph and an RGB photo; a
pooled warden inspects both; the sender chooses carriers to stay unnoticed.

All numbers below come from `docs/RESULTS.md`, which `scripts/make_report.py`
builds from saved experiment outputs. Small scale: 195 photo crops, 12 prompts,
a small offline language model (Hugging Face was unreachable on the build machine).

## What works (built and tested)

| Part | Status | Evidence |
| --- | --- | --- |
| Framing: split, headers, CRC, whitening | done | unit tests |
| Text channel: arithmetic coding + OD-Stega budget + exact KL | done | 1078 / 1078 stego-texts recovered exactly |
| Image channel: LSB replacement, LSB matching, HILL + STC on RGB | done | 2135 / 2135 stego images recovered exactly |
| Sender / receiver, live demo, damage reporting | done | 13 / 13 end-to-end tests; JPEG damage reported as "image fragment lost" |
| Detectors: SPA, 3 pretrained CNNs, SPAM-LR, n-gram, likelihood | done | AUC tables with 95% intervals |
| Pooled warden (p-value fusion, 5% false alarms) | done | used in selection experiment |
| Carrier selection: baselines + PPO (two designs) | done | held-out evaluation, two wardens |

## Findings from the first-stage experiments (measured, not assumed)

1. **A leak in the reused text coder, found and fixed.** subtext-codec sends its
   length field unscrambled, so equal-length messages open with the same words.
   Detectors caught this (held-out AUC 0.96). Adding a random 32-bit nonce and
   whitening the whole bitstream brought the black-box detector to chance
   (AUC 0.52 [0.43, 0.60]).
2. **A structural fingerprint of exact-decoding text steganography.** A detector
   that knows the model still reaches AUC 0.71 at C = 0. One feature explains it:
   40% (67 / 168) of ordinary samples contain a token that re-tokenises outside
   the model's top 64; 0 / 168 stego-texts do, because exact decoding needs a
   clean tokenisation. Replacing the filter guard with OD-Stega's retry approach
   did not change this (AUC 0.71) and doubled encoding cost.
3. **The OD-Stega budget trades capacity for detectability.** C = 0.1 raised
   capacity from 2.65 to 3.29 bits/token but made the white-box detector near
   perfect (AUC 0.99).
4. **A capacity bug in HILL + STC on real photos, found and fixed.** Photos
   with blown-out or perfectly flat regions broke pySTC silently. A per-photo
   capacity rule plus verify-after-embed now refuses such payloads (205 honest
   refusals, 0 broken images).
5. **Image detectability depends strongly on the photo source.** SPA catches
   LSB replacement (AUC 0.73–0.99) and nothing else. Aletheia's pretrained CNNs
   are blind on Kodak (≈ 0.50 everywhere) but detect HILL on the scikit-image
   photos (0.61 / 0.70 / 0.81 at 0.05 / 0.1 / 0.2 bits per value).
6. **Choosing the carriers matters.** A one-line heuristic (busiest photo,
   highest-entropy prompt, C = 0) cut the flag rate from 10% / 4% (no selection)
   to 0% / 0% under both wardens on unseen photos and prompts.
7. **RL has not beaten the heuristic yet, and the experiments show why.** PPO v1
   (pick a slot) barely learned. PPO v2 (score candidates) exploited a blind
   spot of its training warden: it always chose C = 0.1, which warden A barely
   notices and warden B catches 100% of the time. Grid search was best under A; on
   unseen Kodak photos under warden B its flag rate is 0.00 vs 0.04 for no
   selection, but mean suspicion is about equal (0.80 vs 0.81), so its advantage
   does not show up in suspicion.

## What is new here, honestly stated

- Splitting a secret across carriers is **not** new (batch steganography,
  multimodal allocation papers).
- What this implementation adds: a **text + image** system judged by a **pooled warden
  with two independent detector sets**, and measurements of three concrete
  failure modes (clear length field, tokenisation fingerprint, proxy-detector
  overfitting). Whether these are new to the literature still needs a proper
  search before any claim.
- The RL contribution is **open**: the first-stage results show the problem is real
  (selection helps) and hard (learned policies overfit the training warden).

## Limitations

- Text numbers use a small offline model, not GPT-2. Re-run with
  `exp_text.py --model gpt2` on the laptop.
- 256×256 crops from two small photo sources; pretrained CNNs were trained on
  512×512 ALASKA2 photos, so they are mismatched by design.
- Selection uses cached measurements with linear interpolation between payload
  rates; final claims need fresh end-to-end encodes.
- No robustness to editing or recompression: both carriers are exact-bit schemes.


