"""
data/prepare_sft.py: Source-agnostic conversational ingestion engine.

Downloads data shards locally via Hugging Face Hub to prevent stream deadlocks,
handles both local files and remote .parquet/.jsonl shards, normalizes schemas
to ChatML via modular format adapters, and partitions into train/val/test splits.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterator, List, Optional

# Ensure repository root is on sys.path regardless of execution context
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
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
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "")).strip()
            content = str(turn.get("content", "")).strip()
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
            if not isinstance(turn, dict):
                continue
            raw_role = str(turn.get("from", "")).strip()
            content = str(turn.get("value", "")).strip()
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
        instruction = str(sample.get("instruction", "")).strip()
        context_input = str(sample.get("input", "")).strip()
        response = str(sample.get("output", "")).strip()
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


def assign_split_hash(text: str, val_ratio: float = 0.05, test_ratio: float = 0.05) -> str:
    """Deterministically assigns sample to 'train', 'val', or 'test' via SHA256 prefix hashing."""
    hash_val = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    score = (hash_val % 100_000) / 100_000.0
    if score < test_ratio:
        return "test"
    elif score < (test_ratio + val_ratio):
        return "val"
    return "train"


def _stream_from_local_file(path: str) -> Iterator[Dict[str, Any]]:
    if path.endswith((".jsonl", ".jsonl.gz")):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                yield from data
    elif path.endswith(".parquet"):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=512):
            yield from batch.to_pylist()


def _resolve_hf_shards(repo: str, subset: Optional[str], token: Optional[str]) -> List[str]:
    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="dataset")
        valid_files = [f for f in files if f.endswith((".parquet", ".jsonl", ".jsonl.gz"))]
        if not valid_files:
            return []

        if subset:
            matches = [f for f in valid_files if subset in f]
            if matches:
                return sorted(matches)

        parquets = [f for f in valid_files if f.endswith(".parquet")]
        return sorted(parquets) if parquets else sorted(valid_files)
    except Exception as e:
        print(f"[Loader] Error querying repository '{repo}': {e}")
        return []


def stream_source(source: str, config: Optional[str] = None, split: str = "train") -> Iterator[Dict[str, Any]]:
    source_path = Path(source)
    if source_path.exists():
        print(f"[Loader] Ingesting local source: {source}")
        yield from _stream_from_local_file(str(source_path))
        return

    # Parse multi-subset list
    if config and "," in config:
        subsets = [c.strip() for c in config.split(",") if c.strip()]
    elif config and config != "all":
        subsets = [config.strip()]
    elif source == "HuggingFaceTB/smoltalk":
        subsets = ["everyday-conversations", "smol-magpie-ultra"]
    else:
        subsets = [config] if config else [None]

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    for subset in subsets:
        subset_name = subset or "default"
        print(f"[Loader] Resolving shard for '{source}' (subset: {subset_name})...")
        shards = _resolve_hf_shards(source, subset, token)

        if not shards:
            print(f"[Warning] No matching shards found for subset '{subset_name}' in {source}.")
            continue

        target_shard = shards[0]
        print(f"[Loader] Downloading shard: {target_shard}")
        local_cached_path = hf_hub_download(
            repo_id=source,
            filename=target_shard,
            repo_type="dataset",
            token=token,
        )
        file_size_mb = os.path.getsize(local_cached_path) / (1024 * 1024)
        print(f"[Loader] Cached locally: {os.path.basename(local_cached_path)} ({file_size_mb:.1f} MB)")

        yield from _stream_from_local_file(local_cached_path)


def prepare_dataset(
    source: str,
    config: Optional[str] = None,
    output_dir: str = "data/sft",
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    max_samples: Optional[int] = None,
    split: str = "train",
):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and partition conversational SFT data.")
    parser.add_argument("--source", type=str, required=True, help="Hugging Face repo or local file path")
    parser.add_argument("--config", type=str, default=None, help="Dataset config or comma-separated subsets")
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
    )