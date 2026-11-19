# Code walkthrough: what we took, what we changed, where

Open these files side by side in VS Code. Line numbers refer to the current
version of the repository.

## 1. The algorithm we started from

**Arithmetic-coding linguistic steganography** (Ziegler, Deng and Rush,
"Neural Linguistic Steganography", EMNLP 2019), using the open-source
implementation **subtext-codec** (MIT licence).

**Idea in one sentence:** at every step the language model gives probabilities
for the next word; the secret bits, read as a number, pick which word comes next.
The receiver runs the same model, sees which word was picked, and gets the bits
back.

| Step | Original (subtext-codec) | Ours (`src/cmstego/text_channel.py`) |
| --- | --- | --- |
| Turn data into bits | `_frame_payload` (codec.py line 424): length and CRC **in clear** + data | `_payload_bits`, **line 183**: random 32-bit nonce + frame **XORed with a keystream** |
| Choose the word candidates | `_step_distribution` (codec.py line 523): top-k + stability filter | `step_distribution`, **line 257**: same filter, **plus** our line-break ban (line 271) and the OD-Stega budget (line 284) |
| Encoding loop: bits choose words | `encode_data_to_text` (codec.py line 702; `reader.decode(cum)` at line 798) | `_encode_once`, **line 302**; `sym = reader.decode(cum)` at **line 320** |
| Decoding loop: words give bits | `decode_text_to_data` (codec.py line 855) | `decode_with_stats`, **line 444**; `writer.encode(...)` at **line 469** |

The original library is installed at
`.venv\Lib\site-packages\subtext_codec\codec.py`.

## 2. What we changed in the text algorithm, and why

### Change 1: nonce + whitening (fixes a leak we found)
- **Problem:** the original puts the message length at the start of the
  bitstream in clear. Messages of the same length therefore start with the same
  words. A detector caught this with AUC 0.99 on GPT-2.
- **Change:** a random 32-bit nonce is sent first; everything after it is XORed
  with SHAKE-256(key, nonce) (`_Keystream`, line 149; `_payload_bits`, line 183).
  The receiver reads the nonce first and rebuilds the same keystream (line 476).
- **Effect:** detector AUC 0.99 → 0.55 (chance), same capacity.

### Change 2: the line-break fix (our own finding)
- **Problem:** GPT-2 writes a blank line as one token, `"\n\n"`. The word after
  it re-reads as `"\n"`, `"\n"`, word, so the stability filter (needed for exact
  decoding) never allows it. Result: 71% of ordinary GPT-2 texts contain a blank
  line, but only 4% of stego-texts. A detector with GPT-2 reached AUC 0.76.
- **Change:**
  - `_newline_sets` / `_newline_banned` (lines 227 and 239): tokens containing a
    blank line, or a line break right after another, are removed from the
    alphabet. This depends only on the context, so the receiver removes the same
    tokens.
  - In `step_distribution`, lines 271–273:
    ```python
    if newline_fix:
        banned = _newline_banned(tokenizer, context_ids)
        alphabet = [t for t in alphabet if t not in banned]
    ```
  - `show_text` (line 247) doubles every line break before the text is sent;
    `unshow_text` (line 252) halves them again at the receiver.
- **Effect (GPT-2, 96 test pairs):** blank lines 75% vs 71% (matching); style
  detector 0.67 → 0.50; white-box detector 0.76 → 0.60, catching 3% at a 5%
  false-alarm rate (chance level); capacity cost under 1%.

### Change 3: OD-Stega budget (integrated from Huang et al., EACL 2026)
- `od_temperature` (line 201) finds the temperature that spends a KL budget
  `C x entropy`; used in `step_distribution` at line 284. We measured the
  trade-off: C = 0.1 gives 29% more capacity but is always caught, so C = 0.

### Change 4: measured capacity instead of guessing (speed)
- `CapacityExceeded` (line 164) reports how many bits fitted when the text ran
  out of tokens; `_fast_fill` (`system.py` line 145) sizes the next attempt from
  it. 1–2 encodes instead of 6–12: sending about 5× faster.

### Change 5: natural ending
- `_finish_sentence` (line 355): after the hidden bits, ordinary sampling until
  the sentence ends. The receiver stops reading once it has the payload.

## 3. Image side (`src/cmstego/image_channel.py`)

Started from: **HILL costs** (Li et al., ICIP 2014) + **syndrome-trellis codes**
(Filler et al., 2011, pySTC library).

| Change | Where | Why |
| --- | --- | --- |
| Capacity rule on *usable* values (not 0/255, not flat) | `usable_values` line 69, `capacity_bits` line 76 | pySTC failed silently on blown-out or flat photos |
| 1024-value header region for the STC length | line 160, used in `_embed_hill` line 168 | pySTC's own 64-value header could not be embedded on such photos |
| Verify after embed | `embed`, line 375 | never output an image that cannot be read |
| **Robust prototype** (new): DCT-domain QIM + Reed-Solomon | `_embed_robust` line 289, `_extract_robust` line 337, `repair_impulses` line 322 | survive JPEG, noise, salt & pepper (see RESULTS.md, robustness) |

## 4. System (`src/cmstego/`)

| Part | Where | What |
| --- | --- | --- |
| All-or-nothing package (Rivest 1997) + compression | `aont.py`: `compress` line 75, `pack` line 91, `unpack` line 103 | either carrier alone reveals nothing |
| Split and rejoin with CRC | `framing.py`: `split_message` line 87, `join_fragments` line 118 | reports which carrier failed |
| Sender / receiver | `system.py`: `send` line 72, `receive` line 120 | ties everything together |

## 5. Our improvement, in three sentences

We attacked arithmetic-coding LLM steganography with detectors and found two
weaknesses. The implementation sent the message length unscrambled (fixed with a
nonce and whitening: 0.99 → 0.55), and exact decoding removes blank lines from the
text (fixed by banning the blank-line token and doubling line breaks for display:
white-box detector 0.76 → 0.60, style detector 0.67 → 0.50). On the image side we
made HILL + STC reliable on real photos and added a robust mode that survives
JPEG re-saving and noise, at a measured cost in detectability.

## 6. Robustness (sir's question about noise and compression)

| Damage | What it is | HILL + STC (default) | Robust prototype |
| --- | --- | --- | --- |
| JPEG re-saving | chat apps compress photos | lost (0%) | survives (100% at 95/90, 96% at 75) |
| Gaussian noise | tiny random change to every pixel | lost (0%) | survives (100% up to σ 2) |
| Salt & pepper | a few pixels turned black/white | lost (0%) | survives (92–96% at 1%) |
| Median filter | smoothing | lost (0%) | lost (8%) |
| Trained detector (SPAM) | how easily it is found | AUC 0.50 (invisible) | AUC 1.00 (always found) |

How the robust prototype works (`image_channel.py`, from line 200):
1. Take the brightness of the photo and cut it into JPEG's own 8×8 blocks.
2. Transform each block with the DCT (the same maths JPEG uses) and pick 5
   low-frequency numbers per block, the parts compression keeps.
3. Each secret bit moves one of these numbers onto one of two grids, 28 apart
   (QIM). A change smaller than 7 cannot flip the bit.
4. The key shuffles the order of the blocks; bits fill one block's 5 numbers before
   moving to the next, so a short secret touches only the blocks it needs
   (`_robust_key`, line 248). Reed-Solomon codes add repair data, so the few bits
   that do flip are corrected.
5. The receiver first decodes as received; if that fails it removes salt & pepper
   dots (pure black/white pixels that differ from their neighbours) and tries again.

### Can it be robust *and* hidden? (measured, `scripts/exp_robust_sweep.py`)

200-byte image share, 144 Kodak crops for detection, 48 for survival:

| Setting | Pixel values changed | PSNR | JPEG 75 | Salt & pepper 1% | SPAM AUC |
|---|---|---|---|---|---|
| Spread over all blocks, grid 28 (old) | 73% | 43.2 dB | 100% | 83% | 1.00 |
| **Packed, grid 28 (new default)** | **47%** | 43.3 dB | **100%** | **92%** | 1.00 |
| Packed, grid 20 | 43% | 46.2 dB | 92% | 92% | 1.00 |
| Packed, grid 14 | 37% | 49.1 dB | 0% | 63% | 1.00 |

Smaller secrets (packed, grid 28): 96 bytes → 25% changed, AUC 0.99;
32 bytes → 17% changed, AUC 0.96 (87% caught at 5% false alarms), still surviving JPEG 75.

What this means: packing is a free improvement (fewer changes, same or better
survival), so it is the default. A smaller grid makes changes smaller but breaks
JPEG survival before it helps hiding. A trained detector still finds every robust
image: robustness needs strong changes, and strong changes leave a trace. Robust
*and* undetectable is not solved here; it is an open trade-off, as in the
literature on robust steganography.
Note: robust images made before this change cannot be read by the new version.

The default (HILL + STC) detects damage with CRC checks and reports which carrier
failed (`framing.py` `join_fragments`, line 118); it never returns wrong data.
