"""Sender and receiver run in different processes. Their logits must match bit for bit,
or decoding breaks. This test encodes in two fresh processes and compares."""

import os
import subprocess
import sys

import pytest

from conftest import DEFAULT_LM, ROOT

pytestmark = pytest.mark.skipif(not (os.path.isdir(DEFAULT_LM) or "/" not in DEFAULT_LM),
                                reason="no language model available")

SCRIPT = f"""
import sys; sys.path.insert(0, {os.path.join(ROOT, 'src')!r})
from cmstego import text_channel as tc
m, t = tc.load_lm({DEFAULT_LM!r})
cfg = tc.TextConfig({DEFAULT_LM!r}, "The history of the", C=0.05)
text, _ = tc.encode(b"determinism across processes", cfg, m, t)
sys.stdout.write(text)
"""


def run():
    return subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=True).stdout


def test_same_text_in_two_processes():
    a, b = run(), run()
    assert a and a == b
