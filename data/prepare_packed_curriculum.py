"""
data/prepare_packed_curriculum.py: High-Throughput Physical Token Interleaving.
Features:
  - Multi-threaded batch tokenization across available CPU cores (GIL-free Rust tiktoken).
  - Background asynchronous shard prefetching (zero download stall pauses).
  - Direct contiguous uint16 memory buffering (eliminates Python heap GC thrashing).
  - Parameter-aware repetition capping across finite reasoning pools.
"""

import concurrent.futures
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


# =====================================================================
# 1. Dynamic Curriculum Weight & Repetition Capping
# =====================================================================

BASE_RATIOS = {
    "FineWeb-Edu": 0.480,
    "DCLM-Edu": 0.250,
    "The Stack-Edu": 0.150,
    "NuminaMath-CoT": 0.050,
    "OpenMathReasoning": 0.040,
    "SLM-Synthetic-Pretrain": 0.030,
}

FINITE_POOLS = {
    "NuminaMath-CoT": 220_000_000,
    "OpenMathReasoning": 200_000_000,
    "SLM-Synthetic-Pretrain": 250_000_000,
}

INFINITE_SOURCES = ["FineWeb-Edu", "DCLM-Edu", "The Stack-Edu"]


def compute_curriculum_weights(total_tokens: int, param_count: int) -> Dict[str, float]:
    """Calculates source weights enforcing safe repetition ceilings on finite pools."""
    if param_count < 200_000_000:
        max_epochs = {"NuminaMath-CoT": 2.0, "OpenMathReasoning": 2.0, "SLM-Synthetic-Pretrain": 1.5}
    elif param_count <= 600_000_000:
        max_epochs = {"NuminaMath-CoT": 3.2, "OpenMathReasoning": 3.0, "SLM-Synthetic-Pretrain": 1.8}
    else:
        max_epochs = {"NuminaMath-CoT": 2.5, "OpenMathReasoning": 2.5, "SLM-Synthetic-Pretrain": 2.0}

    finite_tokens = {}
    for name, pool in FINITE_POOLS.items():
        desired = int(total_tokens * BASE_RATIOS[name])
        ceiling = int(pool * max_epochs[name])
        finite_tokens[name] = min(desired, ceiling)

    rem_tokens = total_tokens - sum(finite_tokens.values())
    inf_sum = sum(BASE_RATIOS[name] for name in INFINITE_SOURCES)

    weights = {}
    for name, count in finite_tokens.items():
        weights[name] = count / total_tokens

    for name in INFINITE_SOURCES:
        share = (BASE_RATIOS[name] / inf_sum) * rem_tokens
        weights[name] = share / total_tokens

    return weights


# =====================================================================
# 2. Fast Batch Tokenizer Helper (Multi-Threaded)
# =====================================================================

def encode_text_batch(tokenizer: Any, texts: List[str]) -> List[List[int]]:
    """Encodes a batch of strings, leveraging multi-threading across all CPU cores."""
    if not texts:
        return []

    # 1. Native wrapper batch encode
    if hasattr(tokenizer, "encode_batch"):
        try:
            return tokenizer.encode_batch(texts)
        except TypeError:
            pass

    # 2. Tiktoken underlying encoder (releases GIL in Rust)
    for attr in ("enc", "_enc", "encoder", "_encoder"):
        underlying = getattr(tokenizer, attr, None)
        if underlying and hasattr(underlying, "encode_batch"):
            num_threads = min(32, os.cpu_count() or 8)
            try:
                return underlying.encode_batch(texts, num_threads=num_threads)
            except Exception:
                break

    # 3. ThreadPool fallback
    workers = min(16, os.cpu_count() or 4)
    if len(texts) > 16 and workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(tokenizer.encode, texts))

    return [tokenizer.encode(t) for t in texts]


# =====================================================================
# 3. Stream Readers with Background Prefetching
# =====================================================================

class SourceReader:
    """Base reader interface for curriculum sources."""
    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight

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
        self._target_files: Optional[List[str]] = None

    def _resolve_target_files(self, token: Optional[str]) -> List[str]:
        api = HfApi(token=token)
        try:
            files = api.list_repo_files(repo_id=self.repo, repo_type="dataset")
            valid_files = [f for f in files if f.endswith((".parquet", ".jsonl", ".jsonl.gz"))]
            if not valid_files:
                return []

            if self.subset:
                valid_files = [f for f in valid_files if self.subset in f]

            parquets = [f for f in valid_files if f.endswith(".parquet")]
            candidates = parquets if parquets else valid_files
            candidates.sort()
            return candidates
        except Exception as e:
            print(f"[{self.name}] Error checking files in {self.repo}: {e}")
            return []

    def _stream_from_jsonl(self, path: str, tokenizer) -> Iterator[List[int]]:
        opener = gzip.open if path.endswith(".gz") else open
        batch_texts: List[str] = []
        batch_size = 1024

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
                        batch_texts.append(text_val)
                        if len(batch_texts) >= batch_size:
                            for doc in encode_text_batch(tokenizer, batch_texts):
                                if doc:
                                    yield doc
                            batch_texts = []
                except Exception:
                    continue

        if batch_texts:
            for doc in encode_text_batch(tokenizer, batch_texts):
                if doc:
                    yield doc

    def _stream_from_parquet(self, path: str, tokenizer) -> Iterator[List[int]]:
        pf = pq.ParquetFile(path)
        schema_cols = pf.schema_arrow.names

        col_to_use = self.text_key if self.text_key in schema_cols else None
        if not col_to_use:
            for candidate in ["text", "content", "prompt", "code"]:
                if candidate in schema_cols:
                    col_to_use = candidate
                    break
        if not col_to_use:
            col_to_use = schema_cols[0]

        # Read in large 2048 chunks for fast batch tokenization
        for batch in pf.iter_batches(batch_size=2048, columns=[col_to_use]):
            raw_texts = batch[col_to_use].to_pylist()
            valid_texts = [t for t in raw_texts if t and isinstance(t, str) and len(t.strip()) > 0]
            if not valid_texts:
                continue
            for doc in encode_text_batch(tokenizer, valid_texts):
                if doc:
                    yield doc

    def stream_docs(self, tokenizer) -> Iterator[List[int]]:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if self._target_files is None:
            self._target_files = self._resolve_target_files(token)
            if not self._target_files:
                print(f"[{self.name}] No valid data files found in {self.repo}.")
                return
            print(f"[{self.name}] Resolved {len(self._target_files)} shard(s) in {self.repo}")

        # Asynchronously prefetch the next parquet file while tokenizing the current one
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            def _fetch(fname):
                return hf_hub_download(repo_id=self.repo, filename=fname, repo_type="dataset", token=token)

            future = pool.submit(_fetch, self._target_files[0]) if self._target_files else None

            for i, target_file in enumerate(self._target_files):
                try:
                    local_path = future.result() if future else None
                except Exception as e:
                    print(f"[{self.name}] Error downloading shard {target_file}: {e}")
                    local_path = None

                # Kick off next download immediately
                if i + 1 < len(self._target_files):
                    future = pool.submit(_fetch, self._target_files[i + 1])
                else:
                    future = None

                if not local_path:
                    continue

                try:
                    if local_path.endswith((".jsonl", ".jsonl.gz")):
                        yield from self._stream_from_jsonl(local_path, tokenizer)
                    else:
                        yield from self._stream_from_parquet(local_path, tokenizer)
                except Exception as e:
                    print(f"[{self.name}] Error reading shard {local_path}: {e}")


class ReasoningParquetReader(FastParquetReader):
    """Specialized parquet reader that stitches problem/solution pairs into an envelope."""
    def __init__(
        self,
        name: str,
        repo: str,
        subset: Optional[str],
        problem_key: str,
        solution_key: str,
        weight: float,
        wrap_think_tag: bool = False,
    ):
        super().__init__(name=name, repo=repo, subset=subset, text_key=problem_key, weight=weight)
        self.problem_key = problem_key
        self.solution_key = solution_key
        self.wrap_think_tag = wrap_think_tag

    def _stream_from_parquet(self, path: str, tokenizer) -> Iterator[List[int]]:
        pf = pq.ParquetFile(path)
        schema_cols = pf.schema_arrow.names

        prob_col = self.problem_key if self.problem_key in schema_cols else None
        if not prob_col:
            for candidate in ["problem", "question", "prompt", "instruction"]:
                if candidate in schema_cols:
                    prob_col = candidate
                    break

        sol_col = self.solution_key if self.solution_key in schema_cols else None
        if not sol_col:
            for candidate in ["generated_solution", "solution", "response", "answer", "output"]:
                if candidate in schema_cols:
                    sol_col = candidate
                    break

        if not prob_col or not sol_col:
            print(f"[{self.name}] Required columns not found in {path}. Columns: {schema_cols}")
            return

        for batch in pf.iter_batches(batch_size=2048, columns=[prob_col, sol_col]):
            probs = batch[prob_col].to_pylist()
            sols = batch[sol_col].to_pylist()
            docs: List[str] = []
            for p, s in zip(probs, sols):
                if not (p and s and isinstance(p, str) and isinstance(s, str)):
                    continue
                p_clean = p.strip()
                s_clean = s.strip()
                if not p_clean or not s_clean:
                    continue

                if self.wrap_think_tag and "<think>" not in s_clean:
                    sol_formatted = f"<think>\n{s_clean}\n</think>"
                else:
                    sol_formatted = s_clean

                docs.append(f"Problem:\n{p_clean}\n\nSolution:\n{sol_formatted}")

            if docs:
                for doc in encode_text_batch(tokenizer, docs):
                    if doc:
                        yield doc


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


# =====================================================================
# 4. Zero-Copy Contiguous Token Packing Engine
# =====================================================================

def pack_curriculum_to_shards(
    sources: List[SourceReader],
    output_dir: str = "data/pretrain",
    total_token_budget: int = 2_000_000_000,
    shard_size_tokens: int = 25_000_000,
    val_tokens: Optional[int] = None,
    tokenizer_type: str = "tiktoken",
    tokenizer_path: Optional[str] = None,
):
    """Packs multiple sources into contiguous .bin files using a preallocated uint16 buffer."""
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = get_tokenizer(tokenizer_type, tokenizer_path)
    eot_id = tokenizer.eot_id

    if val_tokens is None:
        val_tokens = max(50_000, min(1_000_000, total_token_budget // 200))

    raw_weights = [s.weight for s in sources]
    total_weight = sum(raw_weights)
    target_proportions = [w / total_weight for w in raw_weights]
    num_sources = len(sources)

    print("=" * 75)
    print("      slm-gpt Physical Token Packing Engine (High-Throughput Streamer)")
    print("=" * 75)
    print(f"Total Train Target:  {total_token_budget:,} tokens")
    print(f"Validation Target:   {val_tokens:,} tokens")
    print(f"Shard Size:          {shard_size_tokens:,} tokens ({shard_size_tokens * 2 / (1024**2):.1f} MB in uint16)")
    print(f"Output Directory:    {output_dir}")
    print("Target Curriculum:")
    for s, p in zip(sources, target_proportions):
        print(f"  • {s.name:<25} {p * 100:>5.1f}% ({int(p * total_token_budget):,} tokens)")
    print("=" * 75)

    generators = [s.stream_docs(tokenizer) for s in sources]
    tokens_per_source = [0] * num_sources

    # 1. Harvest validation partition first
    print("\nExtracting validation partition...")
    val_buffer: List[int] = []
    source_idx = 0
    last_val_log = 0

    while len(val_buffer) < val_tokens:
        try:
            doc = next(generators[source_idx % num_sources])
            val_buffer.extend(doc)
            val_buffer.append(eot_id)
        except StopIteration:
            generators[source_idx % num_sources] = sources[source_idx % num_sources].stream_docs(tokenizer)
        source_idx += 1

        if len(val_buffer) - last_val_log >= max(10_000, val_tokens // 5):
            pct = (len(val_buffer) / val_tokens) * 100
            print(f"  [Validation] Packed {len(val_buffer):,} / {val_tokens:,} tokens ({pct:.1f}%)")
            last_val_log = len(val_buffer)

    val_data = np.array(val_buffer[:val_tokens], dtype=np.uint16)
    val_file = os.path.join(output_dir, "val_00000.bin")
    val_data.tofile(val_file)
    print(f"✓ Wrote validation shard: {val_file} ({len(val_data):,} tokens, {os.path.getsize(val_file)/(1024*1024):.2f} MB)")

    # 2. Pack Training Shards using pre-allocated numpy ring buffer
    print("\nPacking training partition...")
    shard_buffer = np.empty(shard_size_tokens, dtype=np.uint16)
    shard_pos = 0
    shard_idx = 0
    total_tokens_written = 0
    last_train_log = 0

    while total_tokens_written + shard_pos < total_token_budget:
        total_so_far = max(1, sum(tokens_per_source))

        # Fast pure-Python deficit calculation
        best_i = 0
        max_deficit = -1e9
        for i in range(num_sources):
            defic = target_proportions[i] - (tokens_per_source[i] / total_so_far)
            if defic > max_deficit:
                max_deficit = defic
                best_i = i
        chosen_idx = best_i

        try:
            doc_tokens = next(generators[chosen_idx])
        except StopIteration:
            generators[chosen_idx] = sources[chosen_idx].stream_docs(tokenizer)
            try:
                doc_tokens = next(generators[chosen_idx])
            except StopIteration:
                continue

        doc_len = len(doc_tokens) + 1
        tokens_per_source[chosen_idx] += doc_len

        # Fast numpy assembly of doc + eot
        doc_arr = np.empty(doc_len, dtype=np.uint16)
        doc_arr[:-1] = doc_tokens
        doc_arr[-1] = eot_id

        doc_offset = 0
        while doc_offset < doc_len:
            space = shard_size_tokens - shard_pos
            chunk_size = min(space, doc_len - doc_offset)
            shard_buffer[shard_pos : shard_pos + chunk_size] = doc_arr[doc_offset : doc_offset + chunk_size]
            shard_pos += chunk_size
            doc_offset += chunk_size

            # Shard full: write directly from contiguous memory
            if shard_pos == shard_size_tokens:
                out_file = os.path.join(output_dir, f"train_{shard_idx:05d}.bin")
                shard_buffer.tofile(out_file)
                total_tokens_written += shard_size_tokens
                shard_idx += 1
                shard_pos = 0
                pct = (total_tokens_written / total_token_budget) * 100
                print(f"✓ Wrote {out_file} | Total: {total_tokens_written:,} / {total_token_budget:,} tokens ({pct:.1f}%)")

                if total_tokens_written >= total_token_budget:
                    break

        total_current = total_tokens_written + shard_pos
        if total_current - last_train_log >= max(50_000, total_token_budget // 20):
            pct = (total_current / total_token_budget) * 100
            print(f"  [Training] Packed {total_current:,} / {total_token_budget:,} tokens ({pct:.1f}%)")
            last_train_log = total_current

    # Final remainder shard
    if shard_pos > 0 and total_tokens_written < total_token_budget:
        out_file = os.path.join(output_dir, f"train_{shard_idx:05d}.bin")
        shard_buffer[:shard_pos].tofile(out_file)
        total_tokens_written += shard_pos
        print(f"✓ Wrote final remainder shard: {out_file} | Total: {total_tokens_written:,} tokens ({os.path.getsize(out_file)/(1024*1024):.2f} MB)")

    print("=" * 75)
    print(f"Physical packing complete. Total train tokens: {total_tokens_written:,}")
    print("=" * 75)


def build_default_curriculum(
    total_tokens: int = 2_000_000_000,
    param_count: int = 126_758_400,
    upstream_dir: Optional[str] = None
) -> List[SourceReader]:
    if upstream_dir and os.path.exists(upstream_dir):
        return [
            LocalBinReader("Local Curated Shards", upstream_dir, weight=0.98, eot_token_id=50256, dtype=np.uint16),
            FastParquetReader("SLM-Synthetic-Pretrain", "tohio/slm-synthetic-pretrain", None, "text", weight=0.02),
        ]

    w = compute_curriculum_weights(total_tokens=total_tokens, param_count=param_count)

    return [
        FastParquetReader("FineWeb-Edu", "HuggingFaceFW/fineweb-edu", "sample/10BT", "text", weight=w["FineWeb-Edu"]),
        FastParquetReader("DCLM-Edu", "HuggingFaceTB/dclm-edu", None, "text", weight=w["DCLM-Edu"]),
        FastParquetReader("The Stack-Edu", "HuggingFaceTB/smollm-corpus", "python-edu", "content", weight=w["The Stack-Edu"]),
        ReasoningParquetReader("OpenMathReasoning", "nvidia/OpenMathReasoning", None, "problem", "generated_solution", weight=w["OpenMathReasoning"], wrap_think_tag=False),
        ReasoningParquetReader("NuminaMath-CoT", "AI-MO/NuminaMath-CoT", None, "problem", "solution", weight=w["NuminaMath-CoT"], wrap_think_tag=True),
        FastParquetReader("SLM-Synthetic-Pretrain", "tohio/slm-synthetic-pretrain", None, "text", weight=w["SLM-Synthetic-Pretrain"]),
    ]


if __name__ == "__main__":
    tokens_target = 2_000_000_000
    param_target = 126_758_400
    shard_size = 25_000_000
    custom_dir = None
    target_out = "data/pretrain"
    val_budget = None

    for arg in sys.argv[1:]:
        if arg.startswith("total_tokens="):
            tokens_target = int(arg.split("=")[1])
        elif arg.startswith("params=") or arg.startswith("param_count="):
            param_target = int(arg.split("=")[1])
        elif arg.startswith("shard_size=") or arg.startswith("shard_size_tokens="):
            shard_size = int(arg.split("=")[1])
        elif arg.startswith("upstream_dir="):
            custom_dir = arg.split("=")[1]
        elif arg.startswith("output_dir="):
            target_out = arg.split("=")[1]
        elif arg.startswith("val_tokens="):
            val_budget = int(arg.split("=")[1])

    curriculum = build_default_curriculum(
        total_tokens=tokens_target,
        param_count=param_target,
        upstream_dir=custom_dir,
    )

    pack_curriculum_to_shards(
        sources=curriculum,
        output_dir=target_out,
        total_token_budget=tokens_target,
        shard_size_tokens=shard_size,
        val_tokens=val_budget,
    )