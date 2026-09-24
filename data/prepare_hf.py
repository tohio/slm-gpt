import argparse
import hashlib
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

# Silence cosmetic multiprocessing tracker warnings on macOS
warnings.filterwarnings("ignore", category=UserWarning, module=".*resource_tracker.*")

load_dotenv()
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tokenizer import PretrainedTiktokenTokenizer


def assign_split_hash(text: str, val_ratio: float, test_ratio: float) -> str:
    hash_val = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    score = (hash_val % 100_000) / 100_000.0
    if score < test_ratio:
        return "test"
    elif score < (test_ratio + val_ratio):
        return "val"
    return "train"


def prepare_hf_stream(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    text_column: str = "text",
    output_dir: str = "data/processed",
    val_ratio: float = 0.01,
    test_ratio: float = 0.01,
    max_tokens: Optional[int] = None,
    buffer_flush_size: int = 100_000,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        print("✓ Authenticated with Hugging Face token from .env")

    print(f"Opening streaming connection to '{dataset_name}'...")
    ds = load_dataset(
        dataset_name,
        dataset_config,
        split="train",
        streaming=True,
        token=hf_token,
    )

    tokenizer = PretrainedTiktokenTokenizer("gpt2")
    eot_token = tokenizer.encode("<|endoftext|>")[0]

    split_paths = {
        "train": out_path / "train.bin",
        "val": out_path / "val.bin",
        "test": out_path / "test.bin",
    }
    for p in split_paths.values():
        with open(p, "wb") as f:
            pass

    buffers: Dict[str, list] = {"train": [], "val": [], "test": []}
    token_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    def flush_buffer(split_name: str):
        buf = buffers[split_name]
        if not buf:
            return
        arr = np.array(buf, dtype=np.uint16)
        with open(split_paths[split_name], "ab") as f:
            f.write(arr.tobytes())
        token_counts[split_name] += len(buf)
        buf.clear()

    total_tokens_target = max_tokens if max_tokens else float("inf")
    pbar = tqdm(total=max_tokens if max_tokens else None, unit="tokens", desc="Streaming & Tokenizing")

    doc_count = 0
    dtype_bytes = np.dtype(np.uint16).itemsize

    try:
        for example in ds:
            text = example.get(text_column, "")
            if not text or not text.strip():
                continue

            target_split = assign_split_hash(text, val_ratio, test_ratio)
            ids = tokenizer.encode(text)
            ids.append(eot_token)

            buffers[target_split].extend(ids)
            pbar.update(len(ids))
            doc_count += 1

            if len(buffers[target_split]) >= buffer_flush_size:
                flush_buffer(target_split)

            current_total = sum(token_counts.values()) + sum(len(b) for b in buffers.values())
            if current_total >= total_tokens_target:
                break
    except KeyboardInterrupt:
        print("\nStreaming paused by user. Flushing remaining buffers to disk...")
    finally:
        for split_name in buffers:
            flush_buffer(split_name)
        pbar.close()

    total_written = sum(token_counts.values())
    print("\nDataset preparation complete:")
    print(f"Total documents processed: {doc_count:,}")
    print(f"Total tokens written:      {total_written:,}")
    for split_name, count in token_counts.items():
        pct = (count / total_written * 100) if total_written > 0 else 0
        mb = (count * dtype_bytes) / (1024 * 1024)
        print(f"  -> {split_name:<5}.bin: {count:>10,} tokens ({pct:>5.1f}%) | {mb:>7.2f} MB")

    sys.stdout.flush()
    # Force hard-exit past lingering background HTTP socket thread loops
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream HF dataset directly to memory-mapped .bin files")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset path")
    parser.add_argument("--config", type=str, default=None, help="HF dataset sub-config")
    parser.add_argument("--text_column", type=str, default="text", help="Text column name")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--test_ratio", type=float, default=0.01)
    parser.add_argument("--max_tokens", type=int, default=None, help="Max token limit")
    parser.add_argument("--buffer_flush_size", type=int, default=200_000, help="Tokens before buffer flush")

    args = parser.parse_args()
    prepare_hf_stream(
        dataset_name=args.dataset,
        dataset_config=args.config,
        text_column=args.text_column,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_tokens=args.max_tokens,
        buffer_flush_size=args.buffer_flush_size,
    )