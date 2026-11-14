# Cross-modal steganography with LLMs and RL

This project splits one secret between a paragraph of GPT-2-generated text and an
RGB image, where the image part is embedded with HILL costs and syndrome-trellis
codes (STC). The receiver recovers the secret exactly from the two carriers. A
pooled warden inspects both carriers together and decides whether the pair looks
suspicious. Choosing which photo, prompt and text-coding budget to use (carrier
selection) is compared across simple baselines and PPO.

## How it works

1. **Seal, split and frame.** The secret is first sealed into an all-or-nothing
   package (Rivest's package transform, built from SHAKE-256/SHA-256): a fresh
   random key locks it and is mixed into the locked data, so either carrier's
   part alone opens nothing. The package is then cut in two, with at least 16
   bytes in each carrier. The text gets as much as fits in a short passage
   (`max_text_tokens`): the sender offers the most it may take and, if that
   overflows, uses the coder's measured capacity to size a second attempt, so a
   send needs 1-2 text encodes. The image takes the rest. Each fragment gets a
   header with a message id, offset, total length and CRC32, and is whitened by
   XOR with a SHAKE-256 keystream from the shared key. Whitening makes the bits
   look random for the text coder; it is not claimed as encryption-grade
   security.
2. **Text channel.** Arithmetic coding over the language model's next-token
   probabilities turns the fragment into text. The bitstream starts with a random
   32-bit nonce, and the rest is whitened with SHAKE-256 of the key and nonce. An
   OD-Stega budget C trades capacity against detectability.
3. **Image channel.** HILL assigns each colour value a cost, and STC finds a
   low-cost set of ±1 changes that carries the fragment. Every stego image is
   decoded once after embedding (verify after embed), and payloads above a photo's
   capacity are refused.
4. **Receiver.** It decodes the text, extracts the image fragment, checks the CRCs,
   joins the fragments and opens the package. If a carrier is damaged, it reports which one failed.

`docs/EXPLAINER.md` explains each part in plain language.

## Setup

```bash
./setup.sh                          # venv, libraries, pySTC, Aletheia models, Kodak covers, GPT-2
source .venv/bin/activate
CMSTEGO_LM=gpt2 python -m pytest -q
```

## Demo

**Sender.** Hides the secret and writes `demo_out/stego.txt` and
`demo_out/stego.png`:

```bash
python scripts/demo.py --model gpt2 --cover my_cover.png --secret "your message"
```

**Preparing your own photo.** Creates `my_cover.png` from `photo.jpg`, resized to
fit within 512 px:

```bash
python -c "from PIL import Image; im=Image.open('photo.jpg').convert('RGB'); im.thumbnail((512,512)); im.save('my_cover.png')"
```

**Receiver.** Recovers the secret from the saved files:

```bash
python scripts/receive.py --model gpt2
```

Using the wrong key shows that recovery fails:

```bash
python scripts/receive.py --model gpt2 --key wrong-key
```

**Attacks.** The sender script can damage a carrier before receiving, to show the
failure being reported:

```bash
python scripts/demo.py --model gpt2 --attack jpeg   # re-saves the image as JPEG quality 90: image fragment lost
python scripts/demo.py --model gpt2 --attack edit   # changes one word of the text: text fragment lost
```

**Browser app.** Runs the same hide and extract steps in a browser:

```bash
python scripts/app.py --model gpt2
```

Then open http://127.0.0.1:7860.

## Where the metrics are

- `docs/RESULTS.md`: all result tables
- `docs/figures/*.png`: figures
- `results/*.csv`: the saved measurements behind those tables
- `results_gpt2/`: the small GPT-2 run

**Reading AUC:** 0.5 means the detector is guessing (good hiding); 1.0 means the
stego carrier is always caught.

## Key results

Every part works end to end and is tested, and every number in `docs/RESULTS.md`
comes from scripts in this repository, but sample sizes are small: 195 photo crops
and 12 prompts. Start with `docs/SUMMARY.md` (one page) for the findings.

**Exact recovery.** 2135 / 2135 stego images and 1078 / 1078 stego-texts. A further
205 image attempts were refused because the payload exceeded that cover's capacity,
and 2 messages could not be encoded by the retry guard; these were reported as
failures, never sent.

### Image channel

- HILL + STC at 0.05 / 0.1 / 0.2 bits per value gives PSNR 69.0 / 65.7 / 62.3 dB.
- SPA catches LSB replacement on Kodak (AUC 0.73–0.99) and nothing else.
- Aletheia's pretrained CNNs are blind on Kodak (≈ 0.50 everywhere) but detect HILL
  on the scikit-image photos (0.61 / 0.70 / 0.81 at 0.05 / 0.1 / 0.2 bits per value).
- These pretrained detectors were trained on 512×512 ALASKA2 photos and are
  mismatched with our 256×256 crops.

### Text channel (toy language model, not GPT-2)

These AUCs were measured with the small toy language model in `data/toy-lm`, not
GPT-2. Each is a held-out ROC-AUC with a 95% interval.

| Variant | C | Payload bits / token | N-gram detector (black box) | Likelihood detector (white box) |
| --- | --- | --- | --- | --- |
| clear (length field unscrambled) | 0 | 2.44 | 0.96 [0.93, 0.98] | 0.96 [0.93, 0.98] |
| filter (+ nonce; final system) | 0 | 2.65 | 0.52 [0.43, 0.60] | 0.71 [0.62, 0.78] |
| filter (+ nonce; final system) | 0.1 | 3.29 | 0.70 [0.62, 0.78] | 0.99 [0.98, 1.00] |
| retry (+ nonce, retry guard) | 0 | 2.55 | 0.47 [0.33, 0.61] | 0.71 [0.59, 0.82] |

- **Length-field leak, found and fixed.** The nonce and whitening brought the
  black-box detector from 0.96 to chance level (0.52).
- **Tokenisation fingerprint, still present.** A white-box detector reaches 0.71 at
  C = 0. On re-tokenising, 67 / 168 ordinary samples contain a token outside the
  model's top 64, versus 0 / 168 stego-texts. The retry guard did not lower this.

### Carrier selection (text measurements from the toy model)

Flag rates at a 5% false-alarm rate, on 1000 held-out messages with unseen Kodak
photos and unseen prompts. Lower is better for the sender.

| Strategy | Carriers evaluated / message | Flag rate, warden A | Flag rate, warden B (never trained on) |
| --- | --- | --- | --- |
| No selection | 1 | 0.10 [0.08, 0.12] | 0.04 [0.03, 0.06] |
| Random | 1 | 0.16 [0.14, 0.19] | 0.49 [0.47, 0.52] |
| Heuristic | 1 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| Grid search (vs A) | 64 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] |
| PPO v1 (slot-index policy) | 1 | 0.08 [0.07, 0.10] | 0.02 [0.01, 0.03] |
| PPO v2 (scoring policy) | 1 | 0.00 [0.00, 0.01] | 1.00 [1.00, 1.00] |

- **Choosing the carriers matters.** The one-line heuristic (busiest photo,
  highest-entropy prompt, C = 0) cut the flag rate from 10% / 4% to 0% / 0%.
- **PPO has not beaten the heuristic.** PPO v1 barely learned. PPO v2 overfitted to
  warden A: it always chose C = 0.1, which warden A barely notices and warden B
  catches 100% of the time.

### Early GPT-2 numbers (small run)

A small GPT-2 run of 47 of 48 messages (1 encode failure at C=0.1) is saved in
`results_gpt2/`. These are early
numbers and are not in `docs/RESULTS.md`.

| C | Payload bits / token | KL vs top-k sampling (bits/token) |
| --- | --- | --- |
| 0 | 2.37 | 0.099 |
| 0.1 | 3.07 | 0.323 |

Detector AUCs from this run rest on only 12 held-out pairs, so they are not reported
as results.

## Reproducing the experiments

```bash
python scripts/exp_image.py                                         # image channel + detectors (~40 min)
python scripts/exp_text.py --model gpt2 --variant clear             # text: subtext-codec framing as-is
python scripts/exp_text.py --model gpt2 --variant filter --cs 0,0.1 # text: + nonce (final system)
python scripts/exp_text.py --model gpt2 --variant retry --cs 0      # text: + nonce, retry guard
python scripts/diag_text_fingerprint.py --model gpt2                # what the white-box detector sees
python scripts/exp_selection.py                                     # warden + baselines + PPO
python scripts/make_report.py                                       # docs/RESULTS.md + figures
python scripts/bench_send.py --model gpt2                           # sender speed: old vs new split search
```

Without internet, `python scripts/make_toy_lm.py` builds a tiny offline model in
`data/toy-lm`, which is what the numbers in `docs/RESULTS.md` used.

## Repository layout

- `src/cmstego/`: the library (framing, text and image channels, sender and receiver, detectors, warden, carrier selection)
- `scripts/`: experiments, demo, receiver, browser app, report generation and setup helpers
- `tests/`: unit and end-to-end tests
- `docs/`: summary, plain-language explainer, generated results and figures
- `results/`: saved experiment outputs behind `docs/RESULTS.md`
- `data/`: the toy language model and the cover photos
- `vendor/`: third-party code downloaded by `setup.sh` (pySTC, Aletheia)

## Limitations and next steps

**Limitations**

- Sample sizes are small: 195 photo crops, 12 prompts, and 47 of 48 messages (1 encode failure at C=0.1) in the GPT-2 run.
- Text detection results use the small toy language model, not GPT-2, and the
  numbers will differ with GPT-2.
- PPO does not beat the hand-written heuristic, and PPO v2 overfits to warden A.
- The tokenisation fingerprint remains: exact-decoding stego-text never contains
  the re-tokenisation events that ordinary samples sometimes do.
- The covers are 256×256 crops from two small photo sources, and the pretrained CNNs
  are mismatched with them by design.
- Selection uses cached measurements with linear interpolation between payload
  rates; final claims need fresh end-to-end encodes.
- There is no robustness to editing or recompression, because both carriers are
  exact-bit schemes.

**Next steps**

- Full GPT-2 text runs.
- A comparison with another text coder.
- A GPU-trained image detector.
- Better RL.

## Third-party code

- [subtext-codec](https://github.com/shevisj/subtext-codec) (MIT): arithmetic coder, framing, stepper, tokenisation guard.
- [pySTC](https://github.com/daniellerch/pySTC) (MIT, wraps Binghamton's STC C++ code): syndrome-trellis codes.
- [Aletheia](https://github.com/daniellerch/aletheia) (MIT): pretrained EfficientNet detectors; SPA re-implemented from it.
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (MIT): PPO.
- Kodak Lossless True Color Image Suite: cover photos.

Methods follow OD-Stega (Huang et al., EACL 2026), HILL (Li et al., ICIP 2014),
STC (Filler et al., IEEE TIFS 2011), SPA (Dumitrescu et al., 2003) and SPAM
(Pevný et al., IEEE TIFS 2010).
