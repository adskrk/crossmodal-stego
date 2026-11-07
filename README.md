# Cross-modal steganography with LLMs and RL — pilot

One secret is split between a paragraph written by a language model and an RGB
photo. The receiver recovers it exactly. A pooled warden inspects both
carriers; a PPO policy learns which photo, prompt and text-coding budget make
the pair least suspicious.

This is a **pilot**: every part works end to end and is tested, and every
number in `docs/RESULTS.md` comes from scripts in this repo, but sample sizes
are small. See `docs/EXPLAINER.md` for how each part works.

## Quick start

```bash
./setup.sh                      # venv, libraries, pySTC, Aletheia models, Kodak covers, GPT-2
source .venv/bin/activate
CMSTEGO_LM=gpt2 python -m pytest -q
python scripts/demo.py --model gpt2
python scripts/demo.py --model gpt2 --attack jpeg   # shows the image fragment breaking
```

## Reproduce the pilot results

```bash
python scripts/exp_image.py                                         # image channel + detectors (~40 min)
python scripts/exp_text.py --model gpt2 --variant clear             # text: subtext-codec framing as-is
python scripts/exp_text.py --model gpt2 --variant filter --cs 0,0.1 # text: + nonce (final system)
python scripts/exp_text.py --model gpt2 --variant retry --cs 0      # text: + nonce, retry guard
python scripts/diag_text_fingerprint.py --model gpt2                # what the white-box detector sees
python scripts/exp_selection.py                                     # warden + baselines + PPO
python scripts/make_report.py                                       # docs/RESULTS.md + figures
```

Start with `docs/SUMMARY.md` (one page), then `docs/EXPLAINER.md`.

Without internet, `python scripts/make_toy_lm.py` builds a tiny offline model
in `data/toy-lm`, which is what the shipped pilot numbers used (see RESULTS.md).

## Layout

| Path | What it is |
| --- | --- |
| `src/cmstego/framing.py` | split / join, fragment headers, CRC, whitening |
| `src/cmstego/text_channel.py` | arithmetic-coding text stego + OD-Stega budget + exact KL |
| `src/cmstego/image_channel.py` | LSB replacement, LSB matching, HILL + STC on RGB |
| `src/cmstego/system.py` | sender and receiver |
| `src/cmstego/detectors/` | SPA, SPAM, Aletheia EfficientNet; n-gram and likelihood text detectors |
| `src/cmstego/warden.py` | p-value calibration and pooled flagging |
| `src/cmstego/selection/pilot.py` | cached-measurement environment, baselines, PPO env |
| `scripts/` | experiments, demo, setup helpers |
| `tests/` | unit and end-to-end tests |

## Third-party code

- [subtext-codec](https://github.com/shevisj/subtext-codec) (MIT): arithmetic coder, framing, stepper, tokenisation guard.
- [pySTC](https://github.com/daniellerch/pySTC) (MIT, wraps Binghamton's STC C++ code): syndrome-trellis codes.
- [Aletheia](https://github.com/daniellerch/aletheia) (MIT): pretrained EfficientNet detectors; SPA re-implemented from it.
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (MIT): PPO.
- Kodak Lossless True Color Image Suite: cover photos.

Methods follow OD-Stega (Huang et al., EACL 2026), HILL (Li et al., ICIP 2014),
STC (Filler et al., IEEE TIFS 2011), SPA (Dumitrescu et al., 2003) and SPAM
(Pevný et al., IEEE TIFS 2010).
