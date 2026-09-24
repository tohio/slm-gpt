"""
data/prepare_packed_curriculum.py: Physical Token Interleaving with Hybrid Parquet/JSONL Support.
Downloads data shards locally via Hugging Face Hub to prevent stream deadlocks,
handles both .parquet and .jsonl/.jsonl.gz sources, and packs tokens into uint16 .bin shards.
"""

import glob
import gzip
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterator, List, Optional

# Ensure repository root is on sys.path regardless of execution context
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import pyarrow.parquet as pq

from tokenizer.factory import get_tokenizer


class SourceReader:
    """Base reader interface for curriculum sources."""
    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight
        self._logged = False

    def stream_docs(self, tokenizer) -> Iterator[List[int]]:
        raise NotImplementedError


class FastParquetReader(SourceReader):
    """Downloads parquet or jsonl shards locally via HF Hub and streams tokenized documents."""
    def __init__(
        self,
        name: str,
        repo: str,
        subset: Optional[str],
        text_key: str,
        weight: float,
    ):
        super().__init__(name, weight)
        self.repo = repo
        self.subset = subset
        self.text_key = text_key
        self._local_path: Optional[str] = None
        self._resolved: bool = False

    def _resolve_target_file(self, token: Optional[str]) -> Optional[str]:
        api = HfApi(token=token)
        try:
            files = api.list_repo_files(repo_id=self.repo, repo_type="dataset")
            valid_files = [f for f in files if f.endswith((".parquet", ".jsonl", ".jsonl.gz"))]
            if not valid_files:
                return None

            if self.subset:
                subset_matches = [f for f in valid_files if self.subset in f]
                if subset_matches:
                    return subset_matches[0]

            # Prefer parquet if present; otherwise fall back to jsonl
            parquets = [f for f in valid_files if f.endswith(".parquet")]
            return parquets[0] if parquets else valid_files[0]
        except Exception as e:
            print(f"[{self.name}] Error checking files in {self.repo}: {e}")
            return None

    def _ensure_local_file(self) -> Optional[str]:
        if self._resolved:
            return self._local_path

        self._resolved = True
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        auth_str = " (authenticated)" if token else " (unauthenticated)"
        print(f"[{self.name}] Finding data shard in {self.repo}{auth_str}...")

        target_file = self._resolve_target_file(token)
        if not target_file:
            print(f"[{self.name}] No valid data files (.parquet, .jsonl) found in repository.")
            return None

        print(f"[{self.name}] Downloading shard: {target_file}")
        self._local_path = hf_hub_download(
            repo_id=self.repo,
            filename=target_file,
            repo_type="dataset",
            token=token,
        )
        file_size_mb = os.path.getsize(self._local_path) / (1024 * 1024)
        print(f"[{self.name}] Cached locally: {os.path.basename(self._local_path)} ({file_size_mb:.1f} MB)")
        return self._local_path

    def _stream_from_jsonl(self, path: str, tokenizer) -> Iterator[List[int]]:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text_val = None
                    if isinstance(data, str):
                        text_val = data
                    elif isinstance(data, dict):
                        text_val = (
                            data.get(self.text_key)
                            or data.get("text")
                            or data.get("content")
                            or data.get("prompt")
                        )
                        if not text_val and "messages" in data and isinstance(data["messages"], list):
                            text_val = "\n".join(
                                f"{m.get('role', '')}: {m.get('content', '')}"
                                for m in data["messages"]
                                if isinstance(m, dict)
                            )
                    if text_val and isinstance(text_val, str) and len(text_val.strip()) > 0:
                        yield tokenizer.encode(text_val)
                except Exception:
                    continue

    def _stream_from_parquet(self, path: str, tokenizer) -> Iterator[List[int]]:
        pf = pq.ParquetFile(path)
        schema_cols = pf.schema_arrow.names

        col_to_use = self.text_key if self.text_key in schema_cols else None
        if not col_to_use:
            for candidate in ["text", "content", "prompt"]:
                if candidate in schema_cols:
                    col_to_use = candidate
                    break
        if not col_to_use:
            col_to_use = schema_cols[0]

        for batch in pf.iter_batches(batch_size=256, columns=[col_to_use]):
            for text_val in batch[col_to_use].to_pylist():
                if text_val and isinstance(text_val, str) and len(text_val.strip()) > 0:
                    yield tokenizer.encode(text_val)

    def stream_docs(self, tokenizer) -> Iterator[List[int]]:
        local_path = self._ensure_local_file()
        if not local_path:
            return

        try:
            if local_path.endswith((".jsonl", ".jsonl.gz")):
                yield from self._stream_from_jsonl(local_path, tokenizer)
            else:
                yield from self._stream_from_parquet(local_path, tokenizer)
        except Exception as e:
            print(f"[{self.name}] Error reading data file: {e}")


class LocalBinReader(SourceReader):
    """Reads pre-tokenized documents from local .bin files."""
    def __init__(self, name: str, directory: str, weight: float, eot_token_id: int, dtype: np.dtype = np.uint16):
        super().__init__(name, weight)
        self.directory = directory
        self.eot_token_id = eot_token_id
        self.dtype = dtype

    def stream_docs(self, tokenizer) -> Iterator[List[int]]:
        shard_files = sorted(glob.glob(os.path.join(self.directory, "*.bin")))
        if not shard_files:
            return

        for sf in shard_files:
            tokens = np.fromfile(sf, dtype=self.dtype)
            eot_indices = np.where(tokens == self.eot_token_id)[0]
            start = 0
            for idx in eot_indices:
                doc = tokens[start:idx].tolist()
                if doc:
                    yield doc
                start = idx + 1
            if start < len(tokens):
                yield tokens[start:].tolist()


def pack_curriculum_to_shards(
    sources: List[SourceReader],
    output_dir: str = "data/pretrain",
    total_token_budget: int = 100_000_000,
    shard_size_tokens: int = 25_000_000,
    val_tokens: Optional[int] = None,
    tokenizer_type: str = "tiktoken",
    tokenizer_path: Optional[str] = None,
):
    """
    Physically packs multiple sources into contiguous .bin files using token-deficit scheduling.
    Guarantees uint16 dtype, train_*.bin and val_00000.bin naming.
    """
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = get_tokenizer(tokenizer_type, tokenizer_path)
    eot_id = tokenizer.eot_id

    if val_tokens is None:
        val_tokens = max(10_000, min(500_000, total_token_budget // 20))

    raw_weights = [s.weight for s in sources]
    total_weight = sum(raw_weights)
    target_proportions = [w / total_weight for w in raw_weights]

    print("=" * 70)
    print("      slm-gpt Physical Token Packing Engine (Local Parquet/JSONL Cache)")
    print("=" * 70)
    print(f"Total Train Target:  {total_token_budget:,} tokens")
    print(f"Validation Target:   {val_tokens:,} tokens")
    print(f"Shard Size:          {shard_size_tokens:,} tokens ({shard_size_tokens * 2 / (1024**2):.1f} MB in uint16)")
    print(f"Output Directory:    {output_dir}")
    print("Target Curriculum:")
    for s, p in zip(sources, target_proportions):
        print(f"  • {s.name:<30} {p * 100:>5.1f}%")
    print("=" * 70)

    generators = [s.stream_docs(tokenizer) for s in sources]
    tokens_per_source = [0] * len(sources)

    # 1. Harvest validation partition first
    print("\nExtracting validation partition...")
    val_buffer: List[int] = []
    source_idx = 0
    last_val_log = 0

    while len(val_buffer) < val_tokens:
        try:
            doc = next(generators[source_idx % len(sources)])
            val_buffer.extend(doc)
            val_buffer.append(eot_id)
        except StopIteration:
            generators[source_idx % len(sources)] = sources[source_idx % len(sources)].stream_docs(tokenizer)
        source_idx += 1

        if len(val_buffer) - last_val_log >= max(5_000, val_tokens // 5):
            pct = (len(val_buffer) / val_tokens) * 100
            print(f"  [Validation] Packed {len(val_buffer):,} / {val_tokens:,} tokens ({pct:.1f}%)")
            last_val_log = len(val_buffer)

    val_data = np.array(val_buffer[:val_tokens], dtype=np.uint16)
    val_file = os.path.join(output_dir, "val_00000.bin")
    val_data.tofile(val_file)
    print(f"✓ Wrote validation shard: {val_file} ({len(val_data):,} tokens, {os.path.getsize(val_file)/1024:.1f} KB)")

    # 2. Pack Training Shards
    print("\nPacking training partition...")
    token_buffer: List[int] = []
    shard_idx = 0
    total_tokens_written = 0
    last_train_log = 0

    while total_tokens_written + len(token_buffer) < total_token_budget:
        total_so_far = max(1, sum(tokens_per_source))
        deficits = [
            target_proportions[i] - (tokens_per_source[i] / total_so_far)
            for i in range(len(sources))
        ]
        chosen_idx = int(np.argmax(deficits))

        try:
            doc_tokens = next(generators[chosen_idx])
        except StopIteration:
            generators[chosen_idx] = sources[chosen_idx].stream_docs(tokenizer)
            try:
                doc_tokens = next(generators[chosen_idx])
            except StopIteration:
                continue

        token_buffer.extend(doc_tokens)
        token_buffer.append(eot_id)
        tokens_per_source[chosen_idx] += len(doc_tokens) + 1

        total_current = total_tokens_written + len(token_buffer)
        if total_current - last_train_log >= max(20_000, total_token_budget // 10):
            pct = (total_current / total_token_budget) * 100
            print(f"  [Training] Packed {total_current:,} / {total_token_budget:,} tokens ({pct:.1f}%)")
            last_train_log = total_current

        while len(token_buffer) >= shard_size_tokens:
            shard_data = np.array(token_buffer[:shard_size_tokens], dtype=np.uint16)
            out_file = os.path.join(output_dir, f"train_{shard_idx:05d}.bin")
            shard_data.tofile(out_file)

            total_tokens_written += shard_size_tokens
            token_buffer = token_buffer[shard_size_tokens:]
            pct = (total_tokens_written / total_token_budget) * 100
            print(f"✓ Wrote {out_file} | Total: {total_tokens_written:,} / {total_token_budget:,} tokens ({pct:.1f}%)")
            shard_idx += 1

            if total_tokens_written >= total_token_budget:
                break

    # Remainder shard
    if token_buffer and total_tokens_written < total_token_budget:
        shard_data = np.array(token_buffer, dtype=np.uint16)
        out_file = os.path.join(output_dir, f"train_{shard_idx:05d}.bin")
        shard_data.tofile(out_file)
        total_tokens_written += len(shard_data)
        print(f"✓ Wrote final remainder shard: {out_file} | Total: {total_tokens_written:,} tokens ({os.path.getsize(out_file)/(1024*1024):.2f} MB)")

    print("=" * 70)
    print(f"Physical packing complete. Total train tokens: {total_tokens_written:,}")
    print("=" * 70)


def build_default_curriculum(upstream_dir: Optional[str] = None) -> List[SourceReader]:
    if upstream_dir and os.path.exists(upstream_dir):
        return [
            LocalBinReader("Local Curated Shards", upstream_dir, weight=0.98, eot_token_id=50256, dtype=np.uint16),
            FastParquetReader("SLM-Synthetic-Pretrain", "tohio/slm-synthetic-pretrain", None, "text", weight=0.02),
        ]

    return [
        FastParquetReader("FineWeb-Edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "text", weight=0.48),
        FastParquetReader("Cosmopedia v2", "HuggingFaceTB/cosmopedia-v2", "default", "text", weight=0.20),
        FastParquetReader("The Stack-Edu", "HuggingFaceTB/smollm-corpus", "python-edu", "content", weight=0.15),
        FastParquetReader("FineMath", "HuggingFaceTB/finemath", "finemath-4+", "text", weight=0.15),
        FastParquetReader("SLM-Synthetic-Pretrain", "tohio/slm-synthetic-pretrain", None, "text", weight=0.02),
    ]


if __name__ == "__main__":
    tokens_target = 100_000_000
    custom_dir = None
    target_out = "data/pretrain"
    val_budget = None

    for arg in sys.argv[1:]:
        if arg.startswith("total_tokens="):
            tokens_target = int(arg.split("=")[1])
        elif arg.startswith("upstream_dir="):
            custom_dir = arg.split("=")[1]
        elif arg.startswith("output_dir="):
            target_out = arg.split("=")[1]
        elif arg.startswith("val_tokens="):
            val_budget = int(arg.split("=")[1])

    curriculum = build_default_curriculum(upstream_dir=custom_dir)
    pack_curriculum_to_shards(
        sources=curriculum,
        output_dir=target_out,
        total_token_budget=tokens_target,
        val_tokens=val_budget,
    )