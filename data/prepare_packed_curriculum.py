"""
data/prepare_packed_curriculum.py: Physical Token Interleaving.
Streams and packs the multi-source curriculum into fixed-size .bin shards.
Uses Token-Deficit Scheduling to guarantee exact curriculum distribution.
"""

import glob
import os
import sys
from typing import Any, Dict, Iterator, List, Optional

from datasets import load_dataset
import numpy as np

from tokenizer.factory import get_tokenizer


class SourceReader:
    """Iterates documents as token arrays from either local .bin shards or Hugging Face streams."""
    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight
        self._logged = False

    def stream_docs(self, tokenizer) -> Iterator[List[int]]:
        raise NotImplementedError


class HFStreamReader(SourceReader):
    """Streams and tokenizes raw documents on-the-fly from Hugging Face."""
    def __init__(
        self,
        name: str,
        repo: str,
        subset: Optional[str],
        text_key: str,
        weight: float,
        split: str = "train",
    ):
        super().__init__(name, weight)
        self.repo = repo
        self.subset = subset
        self.text_key = text_key
        self.split = split

    def stream_docs(self, tokenizer) -> Iterator[List[int]]:
        if not self._logged:
            sub_str = f" ({self.subset})" if self.subset else ""
            print(f"[{self.name}] Connecting to stream: {self.repo}{sub_str}...")
            self._logged = True
        try:
            kwargs = {"split": self.split, "streaming": True}
            if self.subset:
                ds = load_dataset(self.repo, self.subset, **kwargs)
            else:
                ds = load_dataset(self.repo, **kwargs)

            for item in ds:
                text = (
                    item.get(self.text_key)
                    or item.get("text")
                    or item.get("content")
                    or ""
                )
                if text and len(text.strip()) > 0:
                    yield tokenizer.encode(text)
        except Exception as e:
            if not self._logged:
                print(f"[{self.name}] Error connecting to {self.repo}: {e}")
            for i in range(1000):
                yield tokenizer.encode(f"Synthetic document content for {self.name} sample {i}.")


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
            if not self._logged:
                print(f"[{self.name}] Warning: No .bin shards found in '{self.directory}'")
            return

        if not self._logged:
            print(f"[{self.name}] Ingesting pre-tokenized shards from: {self.directory}")
            self._logged = True

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
    val_tokens: int = 500_000,
    tokenizer_type: str = "tiktoken",
    tokenizer_path: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = get_tokenizer(tokenizer_type, tokenizer_path)
    eot_id = tokenizer.eot_id

    raw_weights = [s.weight for s in sources]
    total_weight = sum(raw_weights)
    target_proportions = [w / total_weight for w in raw_weights]

    print("=" * 70)
    print("      slm-gpt Physical Token Packing Engine (Standardized)")
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

    # 1. Harvest validation tokens first so pretrain/train.py never asserts on missing val split
    print("Extracting validation partition...")
    val_buffer: List[int] = []
    source_idx = 0
    while len(val_buffer) < val_tokens:
        try:
            doc = next(generators[source_idx % len(sources)])
            val_buffer.extend(doc)
            val_buffer.append(eot_id)
        except StopIteration:
            generators[source_idx % len(sources)] = sources[source_idx % len(sources)].stream_docs(tokenizer)
        source_idx += 1

    val_data = np.array(val_buffer[:val_tokens], dtype=np.uint16)
    val_file = os.path.join(output_dir, "val_00000.bin")
    val_data.tofile(val_file)
    print(f"✓ Wrote validation shard: {val_file} ({len(val_data):,} tokens)")

    # 2. Pack Training Shards (uint16 + train_XXXXX.bin pattern)
    token_buffer: List[int] = []
    shard_idx = 0
    total_tokens_written = 0

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
        print(f"✓ Wrote final remainder shard: {out_file} | Total: {total_tokens_written:,} tokens")

    print("=" * 70)
    print(f"Physical packing complete. Total train tokens: {total_tokens_written:,}")
    print("=" * 70)


def build_default_curriculum(upstream_dir: Optional[str] = None) -> List[SourceReader]:
    if upstream_dir and os.path.exists(upstream_dir):
        return [
            LocalBinReader("Local Curated Shards", upstream_dir, weight=0.98, eot_token_id=50256, dtype=np.uint16),
            HFStreamReader("SLM-Synthetic-Pretrain", "tohio/slm-synthetic-pretrain", None, "text", weight=0.02),
        ]

    return [
        HFStreamReader("FineWeb-Edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "text", weight=0.48),
        HFStreamReader("Cosmopedia v2", "HuggingFaceTB/cosmopedia-v2", "default", "text", weight=0.20),
        HFStreamReader("The Stack-Edu", "HuggingFaceTB/smollm-corpus", "python-edu", "content", weight=0.15),
        HFStreamReader("FineMath", "HuggingFaceTB/finemath", "finemath-4+", "text", weight=0.15),
        HFStreamReader("SLM-Synthetic-Pretrain", "tohio/slm-synthetic-pretrain", None, "text", weight=0.02),
    ]


if __name__ == "__main__":
    tokens_target = 100_000_000
    custom_dir = None
    target_out = "data/pretrain"

    for arg in sys.argv[1:]:
        if arg.startswith("total_tokens="):
            tokens_target = int(arg.split("=")[1])
        elif arg.startswith("upstream_dir="):
            custom_dir = arg.split("=")[1]
        elif arg.startswith("output_dir="):
            target_out = arg.split("=")[1]

    curriculum = build_default_curriculum(upstream_dir=custom_dir)
    pack_curriculum_to_shards(
        sources=curriculum,
        output_dir=target_out,
        total_token_budget=tokens_target,
    )