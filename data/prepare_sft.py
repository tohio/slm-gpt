"""
data/prepare_sft.py: Source-agnostic conversational ingestion engine.

Supports Hugging Face streaming and local files (.jsonl, .json, .parquet),
normalizes schema to ChatML via modular format adapters, produces train/val/test
splits via deterministic hashing, and exits cleanly.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterator, List, Optional

# Ensure repository root is on sys.path regardless of execution context
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


class OpenAIFormatAdapter:
    @staticmethod
    def match(sample: Dict[str, Any]) -> bool:
        return "messages" in sample and isinstance(sample["messages"], list)

    @staticmethod
    def extract(sample: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        cleaned = []
        for turn in sample.get("messages", []):
            role = turn.get("role", "").strip()
            content = turn.get("content", "").strip()
            if role in ("system", "user", "assistant") and content:
                cleaned.append({"role": role, "content": content})
        return cleaned if len(cleaned) >= 2 else None


class ShareGPTFormatAdapter:
    ROLE_MAP = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "chatgpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }

    @staticmethod
    def match(sample: Dict[str, Any]) -> bool:
        return "conversations" in sample and isinstance(sample["conversations"], list)

    @staticmethod
    def extract(sample: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        cleaned = []
        for turn in sample.get("conversations", []):
            raw_role = turn.get("from", "").strip()
            content = turn.get("value", "").strip()
            role = ShareGPTFormatAdapter.ROLE_MAP.get(raw_role)
            if role and content:
                cleaned.append({"role": role, "content": content})
        return cleaned if len(cleaned) >= 2 else None


class AlpacaFormatAdapter:
    @staticmethod
    def match(sample: Dict[str, Any]) -> bool:
        return "instruction" in sample and "output" in sample

    @staticmethod
    def extract(sample: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        instruction = sample.get("instruction", "").strip()
        context_input = sample.get("input", "").strip()
        response = sample.get("output", "").strip()
        if not instruction or not response:
            return None
        user_prompt = f"{instruction}\n\nContext:\n{context_input}" if context_input else instruction
        return [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ]


ADAPTERS = [OpenAIFormatAdapter, ShareGPTFormatAdapter, AlpacaFormatAdapter]


def normalize_record(record: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """Inspects raw records across known schemas and returns canonical ChatML messages."""
    for adapter in ADAPTERS:
        if adapter.match(record):
            messages = adapter.extract(record)
            if messages:
                roles = {m["role"] for m in messages}
                if "user" in roles and "assistant" in roles:
                    return messages
    return None


# Alias for backwards compatibility with test suite
normalize_messages = normalize_record


def assign_split_hash(text: str, val_ratio: float = 0.05, test_ratio: float = 0.05) -> str:
    """Deterministically assigns sample to 'train', 'val', or 'test' via SHA256 prefix hashing."""
    hash_val = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    score = (hash_val % 100_000) / 100_000.0
    if score < test_ratio:
        return "test"
    elif score < (test_ratio + val_ratio):
        return "val"
    return "train"


def stream_source(source: str, config: Optional[str] = None, split: str = "train") -> Iterator[Dict[str, Any]]:
    """Yields individual raw dialogue records from a local file or Hugging Face dataset stream."""
    source_path = Path(source)
    if source_path.exists() and source_path.is_file():
        print(f"[Loader] Ingesting local file: {source}")
        if source.endswith(".jsonl"):
            with open(source, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        elif source.endswith(".json"):
            with open(source, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    yield item
    else:
        # Automatic fallback for multi-config datasets like SmolTalk
        if source == "HuggingFaceTB/smoltalk" and not config:
            config = "all"

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        print(f"[Loader] Streaming from Hugging Face: '{source}' (config: {config})")
        ds = load_dataset(source, config, split=split, streaming=True, token=hf_token)
        for example in ds:
            yield example


def prepare_dataset(
    source: str,
    config: Optional[str] = None,
    output_dir: str = "data/sft",
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    max_samples: Optional[int] = None,
    split: str = "train",
    exit_on_complete: bool = False,
):
    """Main pipeline to stream, standardize, partition, and write ChatML splits."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = {
        "train": open(out_path / "train.jsonl", "w", encoding="utf-8"),
        "val": open(out_path / "val.jsonl", "w", encoding="utf-8"),
        "test": open(out_path / "test.jsonl", "w", encoding="utf-8"),
    }
    counts = {"train": 0, "val": 0, "test": 0}
    pbar = tqdm(total=max_samples, unit="convs", desc="Processing SFT Data")

    try:
        for idx, raw_record in enumerate(stream_source(source, config=config, split=split)):
            messages = normalize_record(raw_record)
            if not messages:
                continue

            user_content = next((m["content"] for m in messages if m["role"] == "user"), str(idx))
            split_name = assign_split_hash(user_content, val_ratio=val_ratio, test_ratio=test_ratio)

            files[split_name].write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            counts[split_name] += 1
            pbar.update(1)

            if max_samples and sum(counts.values()) >= max_samples:
                break
    finally:
        for f in files.values():
            f.close()
        pbar.close()

    total = sum(counts.values())
    print(f"\nSFT Dataset Preparation Complete ({total:,} dialogues):")
    for s in ("train", "val", "test"):
        pct = (counts[s] / max(1, total)) * 100
        print(f"  -> {s:<5}.jsonl: {counts[s]:>6,} ({pct:>4.1f}%)")

    sys.stdout.flush()
    if exit_on_complete:
        os._exit(0)


# Alias for backwards compatibility with test suite
prepare_sft_dataset = prepare_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and partition conversational SFT data.")
    parser.add_argument("--source", type=str, required=True, help="Hugging Face repo or local file path")
    parser.add_argument("--config", type=str, default=None, help="Dataset config/subset")
    parser.add_argument("--output_dir", type=str, default="data/sft", help="Destination directory")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation partition ratio")
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Test partition ratio")
    parser.add_argument("--max_samples", type=int, default=50_000, help="Maximum samples to process")
    parser.add_argument("--split", type=str, default="train", help="Hugging Face split name")

    args = parser.parse_args()
    prepare_dataset(
        source=args.source,
        config=args.config,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_samples=args.max_samples,
        split=args.split,
        exit_on_complete=True,
    )