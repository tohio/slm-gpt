import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from tokenizer.tiktoken_wrap import PretrainedTiktokenTokenizer


def test_tiktoken_wrapper():
    tok = PretrainedTiktokenTokenizer("gpt2")
    # Update expected vocab size to include the 2 ChatML special tokens
    assert tok.vocab_size == 50259, f"Expected 50259, got {tok.vocab_size}"

    text = "Building a GPT-style transformer from scratch."
    tokens = tok.encode(text)
    decoded = tok.decode(tokens)

    assert decoded == text, f"Roundtrip failed! Expected '{text}', got '{decoded}'"
    print("✓ PretrainedTiktokenTokenizer verified.")


if __name__ == "__main__":
    test_tiktoken_wrapper()
