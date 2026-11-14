#!/usr/bin/env bash
# One-time setup on a Linux/macOS laptop (Python 3.10+, internet access).
# Windows: double-click setup_windows.bat instead.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# CPU-only PyTorch is smaller; fall back to the default wheel if the CPU index is unreachable.
pip install torch --index-url https://download.pytorch.org/whl/cpu || pip install torch
pip install -r requirements.txt

mkdir -p vendor data
# pySTC (MIT): syndrome-trellis codes. Its PyPI package is missing header files, so build from source.
[ -d vendor/pySTC ] || git clone --depth 1 https://github.com/daniellerch/pySTC.git vendor/pySTC
pip install ./vendor/pySTC

# Aletheia (MIT): only the pretrained detector weights we use (~250 MB).
if [ ! -d vendor/aletheia ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/daniellerch/aletheia.git vendor/aletheia
  (cd vendor/aletheia && git sparse-checkout set aletheia-models)
fi

# Cover images: Kodak Lossless True Color Image Suite (24 photos).
[ -d data/kodak ] || git clone --depth 1 https://github.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite.git data/kodak
python scripts/prepare_covers.py --kodak data/kodak/PhotoCD_PCD0992 --out data/covers

# Language model: GPT-2 downloads automatically from Hugging Face on first use.
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('gpt2'); AutoModelForCausalLM.from_pretrained('gpt2'); print('gpt2 ready')"

echo
echo "Setup done. Next:"
echo "  source .venv/bin/activate"
echo "  CMSTEGO_LM=gpt2 python -m pytest -q          # all tests"
echo "  python scripts/demo.py --model gpt2          # live demo"
