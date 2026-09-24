"""
tokenizer/test/test_tiktoken.py: Tests for PretrainedTiktokenTokenizer wrapper.
Validates vocabulary size with slm-gpt canonical special tokens and roundtrip decoding.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from tokenizer.tiktoken_wrap import PretrainedTiktokenTokenizer


def test_tiktoken_wrapper():
    tok = PretrainedTiktokenTokenizer("gpt2")
    # 50,256 base byte-pair ranks + 17 canonical special tokens (50256..50272) = 50,273
    assert tok.vocab_size == 50273, f"Expected 50273, got {tok.vocab_size}"

    text = "Building a GPT-style transformer from scratch."
    tokens = tok.encode(text)
    decoded = tok.decode(tokens)

    assert decoded == text, f"Roundtrip failed! Expected '{text}', got '{decoded}'"
    print("✓ PretrainedTiktokenTokenizer verified.")


if __name__ == "__main__":
    test_tiktoken_wrapper()