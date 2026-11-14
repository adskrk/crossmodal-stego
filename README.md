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

### Windows (no WSL needed)

1. Install **Python 3.11** from
   [python.org](https://www.python.org/downloads/release/python-3119/)
   ("Windows installer (64-bit)"); keep "py launcher" ticked. Windows uses 3.11
   because the prebuilt pySTC library exists only for that version.
2. Install [Git for Windows](https://git-scm.com/download/win) if you do not have it.
3. Clone this repository into a normal folder (not OneDrive), e.g.
   `C:\Users\<you>\Projects\crossmodal-stego`.
4. Double-click **`setup_windows.bat`** (20-40 minutes, downloads about 1.5 GB).
5. Double-click **`run_tests.bat`**: it should end with "passed".
6. Double-click **`run_app.bat`**: the browser demo opens by itself.

To run a script by hand, open a terminal in the project folder and use
`.venv\Scripts\python.exe`, e.g. `.venv\Scripts\python.exe scripts\demo.py --model gpt2`.

### Linux / macOS / WSL

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

**Browser app.** Runs the same hide and extract steps in a browser (on Windows,
just double-click `run_app.bat`):

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

**Exact recovery.** 2135 / 2135 stego images; 288 / 288 GPT-2 stego-texts (and
1078 / 1078 with the toy model). A further 205 image attempts were refused because
the payload exceeded that cover's capacity; these were reported as failures, never
sent.

### Image channel

- HILL + STC at 0.05 / 0.1 / 0.2 bits per value gives PSNR 69.0 / 65.7 / 62.3 dB.
- SPA catches LSB replacement on Kodak (AUC 0.73–0.99) and nothing else.
- Aletheia's pretrained CNNs are at chance on Kodak (≈ 0.50) but detect HILL on
  the scikit-image photos (AUC 0.61 / 0.70 / 0.81; P_E 0.37 / 0.31 / 0.21).
- These pretrained detectors were trained on 512×512 ALASKA2 photos and are
  mismatched with our 256×256 crops.

### Text channel (GPT-2)

Held-out ROC-AUC with 95% intervals, 48 test pairs per row; 0.5 = chance.

| Variant | C | Payload bits / token | N-gram (black box) | Likelihood (white box) | Style (black box) |
| --- | --- | --- | --- | --- | --- |
| clear (length field unscrambled) | 0 | 2.42 | 0.99 [0.97, 1.00] | 0.91 [0.84, 0.96] | 0.71 [0.60, 0.80] |
| filter (+ nonce; final system) | 0 | 2.39 | 0.55 [0.44, 0.66] | 0.76 [0.66, 0.85] | 0.67 [0.57, 0.78] |
| filter (+ nonce; final system) | 0.1 | 3.08 | 0.64 [0.52, 0.74] | 1.00 [1.00, 1.00] | 0.78 [0.68, 0.87] |
| **nlfix (+ line-break fix; default now)** | 0 | 2.35 | **0.51 [0.40, 0.63]** | **0.52 [0.39, 0.63]** | **0.51 [0.40, 0.63]** |

- **Length-field leak, found and fixed.** The nonce and whitening brought the
  n-gram detector from 0.99 to chance level, at the same capacity.
- **Tokenisation fingerprint, found and fixed.** 71% of ordinary GPT-2 texts
  contain a blank line but only 4% of stego-texts, because exact decoding cannot
  allow the blank-line token. With the line-break fix (blank-line tokens banned,
  line breaks shown doubled and halved again by the receiver) all three detectors,
  including the white-box one, are at chance (caught at 5% false alarms: 4% each),
  for 2% less capacity. A weaker trace remains: when re-read, 23 / 96 fixed
  stego-texts contain a token outside the top 64, versus 5 / 96 ordinary texts
  (single-feature AUC 0.59).
- **C = 0.1** buys 29% more capacity but is always caught by the white-box detector.

The toy-model run (more budgets and the retry guard) is in `docs/RESULTS.md`,
section 2b.

### Carrier selection (GPT-2 text measurements)

Flag rates at a 5% false-alarm rate, 1000 held-out messages with unseen Kodak
photos and unseen prompts. Lower is better for the sender. Warden B is never used
in training; the ensemble is wardens A, C and D (no detector shared with B).

| Strategy | Tuned against | Flag rate, own warden(s) | Flag rate, warden B |
| --- | --- | --- | --- |
| No selection | — | A: 0.03 | 0.21 [0.18, 0.23] |
| Heuristic (busiest photo, highest-entropy prompt, C = 0) | — | A: 0.00 | 0.24 [0.21, 0.27] |
| Grid search (64 candidates) | A | 0.00 | 0.49 [0.46, 0.52] |
| Grid search (64 candidates) | ensemble | 0.00 | 0.47 [0.44, 0.50] |
| PPO v1 (slot-index policy) | A | 0.00 | 0.20 [0.18, 0.23] |
| PPO v2 (scoring policy) | A | 0.00 | 0.27 [0.24, 0.30] |
| PPO (scoring policy) | ensemble | 0.11 | 0.16 [0.13, 0.18] |

- **Tuning on known detectors overfits them.** Searches reach about 0% flags on
  the wardens they were tuned on but are caught more by the unseen warden B, even
  when tuned against three wardens at once.
- **PPO vs the ensemble** has the lowest flag rate under B on Kodak, but part of
  its advantage comes from preferring one test prompt whose few held-out texts B
  does not flag (see `docs/SUMMARY.md`), so this is weak evidence.

**Choosing the split.** Letting the policy choose how much of the package goes into
the text (24, 64 or 160 bytes) gave no gain: under the unseen warden B the smallest
text share had the lowest mean suspicion (0.69 vs 0.83–0.88 for every strategy that
chose longer texts), and longer texts are slightly more detectable to the white-box
detector (AUC 0.76 at 81 tokens, 0.81–0.82 at 173–409). Flag rates from this
experiment are not usable: with 24 calibration texts the smallest possible p-value
(0.04) is above the flag threshold (see `docs/RESULTS.md`, section 5).

### Speed (GPT-2, laptop CPU, `scripts/bench_send.py`)

| Secret | Old split search | Capacity-sized search |
| --- | --- | --- |
| 27 bytes | 50.5 s (6 encodes) | 10.2 s (1 encode) |
| 1,000 bytes | 92.6 s (11 encodes) | 16.2 s (2 encodes) |

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
python scripts/add_style_detector.py --results results_gpt2         # third text detector (no model needed)
python scripts/exp_selection.py --results results_gpt2 --ensemble   # selection trained against several wardens
```

On Windows, `run_overnight.bat` runs the GPT-2 measurements for the line-break fix,
the fingerprint check, longer texts and the split experiment; `run_overnight_2.bat`
then measures the final text channel (line-break fix, 16 texts per prompt and
budget) and re-runs the selection experiments on it in `results_gpt2_nlfix/`.

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

- Sample sizes are small: 195 photo crops, 12 prompts, 4 held-out texts per prompt
  and budget in the selection experiments.
- No selection strategy tuned on known detectors has beaten *no selection* under
  the unseen warden yet; they overfit the detectors they were tuned on.
- The line-break fix is measured on 96 messages (48 test pairs); its intervals
  still reach 0.63, and a weaker re-tokenisation trace remains.
- The split experiment needs more texts per size before its flag rates mean
  anything.
- The covers are 256×256 crops from two small photo sources, and the pretrained CNNs
  are mismatched with them by design.
- Selection uses cached measurements with linear interpolation between payload
  rates; final claims need fresh end-to-end encodes.
- There is no robustness to editing or recompression, because both carriers are
  exact-bit schemes.
- The lock-box construction has not had an independent security review.

**Next steps**

- More held-out texts per prompt and per text size, so selection and split
  results do not hinge on a few texts.
- Re-run the selection experiments with the line-break fix on.
- A comparison with another published text coder (e.g. Discop or Meteor).
- A GPU-trained image detector and a second photo dataset.
- Speed: reuse measured capacity, and test a faster inference engine.

## Third-party code

- [subtext-codec](https://github.com/shevisj/subtext-codec) (MIT): arithmetic coder, framing, stepper, tokenisation guard.
- [pySTC](https://github.com/daniellerch/pySTC) (MIT, wraps Binghamton's STC C++ code): syndrome-trellis codes.
- [Aletheia](https://github.com/daniellerch/aletheia) (MIT): pretrained EfficientNet detectors; SPA re-implemented from it.
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (MIT): PPO.
- Kodak Lossless True Color Image Suite: cover photos.

Methods follow OD-Stega (Huang et al., EACL 2026), HILL (Li et al., ICIP 2014),
STC (Filler et al., IEEE TIFS 2011), SPA (Dumitrescu et al., 2003) and SPAM
(Pevný et al., IEEE TIFS 2010).
