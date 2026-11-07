import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# Text-channel tests need a model. Set CMSTEGO_LM to a Hugging Face id (e.g. "gpt2")
# on a machine with internet, or leave the default offline toy model.
DEFAULT_LM = os.environ.get("CMSTEGO_LM", os.path.join(ROOT, "data", "toy-lm"))
