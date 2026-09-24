import argparse
from pathlib import Path
from typing import Tuple
import numpy as np

from tokenizer import PretrainedTiktokenTokenizer, CustomBPETokenizer


def split_text(text: str, val_ratio: float, test_ratio: float) -> Tuple[str, str, str]:
    """Splits raw text sequentially into train, val, and test partitions."""
    assert 0.0 <= val_ratio < 1.0, f"Invalid val_ratio: {val_ratio}"
    assert 0.0 <= test_ratio < 1.0, f"Invalid test_ratio: {test_ratio}"
    assert (val_ratio + test_ratio) < 1.0, (
        f"Sum of val_ratio ({val_ratio}) and test_ratio ({test_ratio}) must be less than 1.0"
    )

    n = len(text)
    train_end = int(n * (1.0 - val_ratio - test_ratio))
    val_end = int(n * (1.0 - test_ratio))

    train_text = text[:train_end]
    val_text = text[train_end:val_end]
    test_text = text[val_end:]

    return train_text, val_text, test_text


def prepare(
    input_file: str,
    output_dir: str,
    tokenizer_type: str = "tiktoken",
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    custom_vocab_size: int = 1024,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = f.read()

    # Step 1: Initialize / Train Tokenizer
    if tokenizer_type == "tiktoken":
        print("Using pre-trained TikToken (gpt2) tokenizer...")
        tokenizer = PretrainedTiktokenTokenizer("gpt2")
    elif tokenizer_type == "custom":
        print(f"Training custom BPE tokenizer to vocab size {custom_vocab_size}...")
        tokenizer = CustomBPETokenizer()
        tokenizer.train(data, vocab_size=custom_vocab_size, verbose=False)
        tokenizer.save(str(out_path / "custom_vocab.json"))
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    # Step 2: Split text
    train_text, val_text, test_text = split_text(data, val_ratio, test_ratio)

    # Step 3: Tokenize each split
    print("Encoding splits into token streams...")
    splits = {
        "train.bin": tokenizer.encode(train_text),
        "val.bin": tokenizer.encode(val_text),
        "test.bin": tokenizer.encode(test_text),
    }

    # Step 4: Serialize to flat binary files
    dtype = np.uint16  # Accommodates vocabulary sizes up to 65,535
    total_tokens = sum(len(ids) for ids in splits.values())

    print(f"\nWriting binary splits to '{output_dir}':")
    for filename, token_ids in splits.items():
        arr = np.array(token_ids, dtype=dtype)
        file_path = out_path / filename
        arr.tofile(file_path)
        pct = (len(arr) / total_tokens * 100) if total_tokens > 0 else 0
        print(
            f"  -> {filename:<10} | {len(arr):>9,} tokens ({pct:>5.1f}%) | "
            f"{arr.nbytes / (1024 * 1024):>6.2f} MB"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare 3-way split text data for transformer training")
    parser.add_argument("--input", type=str, required=True, help="Path to input .txt file")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save .bin files")
    parser.add_argument("--tokenizer", type=str, default="tiktoken", choices=["tiktoken", "custom"])
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation set ratio (e.g., 0.05 = 5%)")
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Test set ratio (e.g., 0.05 = 5%)")
    parser.add_argument("--custom_vocab_size", type=int, default=2048, help="Vocabulary size if using custom tokenizer")

    args = parser.parse_args()
    prepare(
        args.input,
        args.output_dir,
        args.tokenizer,
        args.val_ratio,
        args.test_ratio,
        args.custom_vocab_size,
    )
