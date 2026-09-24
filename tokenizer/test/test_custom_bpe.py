"""
tokenizer/test/test_custom_bpe.py: Unit tests for CustomBPETokenizer.
Tests training, lossless roundtrip decoding, and serialization save/load contracts.
"""

import sys
from pathlib import Path
import tempfile

sys.path.append(str(Path(__file__).resolve().parents[2]))

from tokenizer.bpe import CustomBPETokenizer


def test_bpe_roundtrip_and_training():
    corpus = (
        "The quick brown fox jumps over the lazy dog. "
        "Transformers are powerful neural networks. "
        "Deep learning enables language understanding. 🚀 "
    ) * 10

    tokenizer = CustomBPETokenizer()
    target_vocab = 300  # 256 base bytes + 17 special tokens + 27 merges
    tokenizer.train(corpus, vocab_size=target_vocab)

    assert tokenizer.vocab_size == target_vocab, (
        f"Expected vocab {target_vocab}, got {tokenizer.vocab_size}"
    )

    test_samples = [
        "The quick brown fox",
        "Transformers are powerful!",
        "Handling edge cases like Unicode: café, naïve, 🤖🔥",
    ]

    for sample in test_samples:
        tokens = tokenizer.encode(sample)
        decoded = tokenizer.decode(tokens)
        assert decoded == sample, f"Roundtrip failed! Original: '{sample}', Decoded: '{decoded}'"

    print("✓ CustomBPETokenizer train & lossless roundtrip verified.")


def test_save_and_load():
    corpus = "Machine learning with PyTorch and Python." * 5
    tok1 = CustomBPETokenizer()
    # Must be >= 256 base bytes + 17 special tokens = 273; 280 allocates 7 learned merges
    tok1.train(corpus, vocab_size=280)

    sample = "Machine learning test."
    expected_ids = tok1.encode(sample)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tok1.save(tmp.name)

        tok2 = CustomBPETokenizer()
        tok2.load(tmp.name)

        loaded_ids = tok2.encode(sample)
        assert expected_ids == loaded_ids, "Loaded tokenizer produced different IDs"
        assert tok2.decode(loaded_ids) == sample, "Loaded tokenizer failed decoding"

    print("✓ CustomBPETokenizer serialization (save/load) verified.")


if __name__ == "__main__":
    test_bpe_roundtrip_and_training()
    test_save_and_load()