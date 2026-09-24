import sys
from pathlib import Path
import tempfile
import numpy as np

# Add project root (slm-gpt) to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from data.prepare import split_text, prepare


def test_split_text_boundaries():
    text = "0123456789" * 10  # 100 chars
    val_ratio = 0.10
    test_ratio = 0.20

    train, val, test = split_text(text, val_ratio, test_ratio)

    assert len(train) == 70, f"Expected train len 70, got {len(train)}"
    assert len(val) == 10, f"Expected val len 10, got {len(val)}"
    assert len(test) == 20, f"Expected test len 20, got {len(test)}"
    # Verify no gap or overlap
    assert train + val + test == text, "Splits do not reconstruct original text exactly"
    print("✓ split_text partition boundaries verified.")


def test_prepare_data_end_to_end():
    sample_text = (
        "The attention mechanism is central to decoder transformers. "
        "Each component must be thoroughly unit tested. "
        "Splitting into train, validation, and test ensures rigorous evaluation."
    ) * 20

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_file = tmp_path / "raw.txt"
        input_file.write_text(sample_text, encoding="utf-8")

        out_dir = tmp_path / "processed"
        prepare(
            input_file=str(input_file),
            output_dir=str(out_dir),
            tokenizer_type="tiktoken",
            val_ratio=0.10,
            test_ratio=0.10,
        )

        train_path = out_dir / "train.bin"
        val_path = out_dir / "val.bin"
        test_path = out_dir / "test.bin"

        assert train_path.exists(), "train.bin was not created"
        assert val_path.exists(), "val.bin was not created"
        assert test_path.exists(), "test.bin was not created"

        train_data = np.fromfile(train_path, dtype=np.uint16)
        val_data = np.fromfile(val_path, dtype=np.uint16)
        test_data = np.fromfile(test_path, dtype=np.uint16)

        assert len(train_data) > 0, "train.bin contains 0 tokens"
        assert len(val_data) > 0, "val.bin contains 0 tokens"
        assert len(test_data) > 0, "test.bin contains 0 tokens"
        assert len(train_data) > len(val_data), "Expected train tokens > val tokens"

    print("✓ data.prepare 3-way binary serialization verified.")


if __name__ == "__main__":
    test_split_text_boundaries()
    test_prepare_data_end_to_end()