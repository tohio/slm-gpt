"""
data/prepare_packed_curriculum.py: Multi-Scale Physical Token Interleaving Engine.
Supports dynamic scaling across 125M, 350M, and 1B parameter configurations.
"""

import glob
import gzip
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterator, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import pyarrow.parquet as pq

from tokenizer.factory import get_tokenizer
from data.datacard import create_dataset_card


# =====================================================================
# 1. Multi-Tier Scaling Logic
# =====================================================================

BASE_RATIOS = {
    # Infinite Foundations
    "FineWeb-Edu":           0.420,  # 42.0% (General knowledge & prose)
    "DCLM-Edu":              0.180,  # 18.0% (High-quality discourse)
    "The-Stack-Filtered":    0.160,  # 16.0% (Verified raw Python repo text)

    # Macro Reasoning Backbones (Finite unique pools)
    "NuminaMath-CoT":        0.080,  #  8.0% (~220M unique pool)
    "OpenMathReasoning":     0.060,  #  6.0% (~200M unique pool)

    # Multi-Signal Synthetic Anchor (task_code, tracing, math, restraint)
    "SLM-Synthetic-Pretrain": 0.100, # 10.0% baseline
}

FINITE_POOLS = {
    "NuminaMath-CoT": 220_000_000,
    "OpenMathReasoning": 200_000_000,
}

INFINITE_SOURCES = ["FineWeb-Edu", "DCLM-Edu", "The-Stack-Filtered"]


def compute_curriculum_weights(total_tokens: int, param_count: int) -> Dict[str, float]:
    """
    Dynamically scales the synthetic anchor and finite pools across model sizes:
      - 125M (<200M params):  High synthetic anchor (10-12%) for circuit seeding.
      - 350M (200M-600M params): Moderate anchor (8-10%) for balanced absorption.
      - 1B   (>600M params):   Controlled anchor (4-6%) with macro repos dominating.
    """
    if param_count < 200_000_000:
        synth_ratio = 0.110
        math_cap_mult = 2.2
    elif param_count <= 600_000_000:
        synth_ratio = 0.090
        math_cap_mult = 3.0
    else:
        synth_ratio = 0.050
        math_cap_mult = 2.2

    # Calculate token budgets for finite math pools
    numina_tokens = min(int(total_tokens * BASE_RATIOS["NuminaMath-CoT"]), int(FINITE_POOLS["NuminaMath-CoT"] * math_cap_mult))
    openmath_tokens = min(int(total_tokens * BASE_RATIOS["OpenMathReasoning"]), int(FINITE_POOLS["OpenMathReasoning"] * math_cap_mult))
    synth_tokens = int(total_tokens * synth_ratio)

    fixed_sum = numina_tokens + openmath_tokens + synth_tokens
    rem_tokens = max(0, total_tokens - fixed_sum)

    # Distribute remaining tokens across infinite backbones
    inf_sum = sum(BASE_RATIOS[s] for s in INFINITE_SOURCES)
    weights = {
        "SLM-Synthetic-Pretrain": synth_tokens / total_tokens,
        "NuminaMath-CoT": numina_tokens / total_tokens,
        "OpenMathReasoning": openmath_tokens / total_tokens,
    }

    for name in INFINITE_SOURCES:
        share = (BASE_RATIOS[name] / inf_sum) * rem_tokens
        weights[name] = share / total_tokens

    return weights


# =====================================================================
# 2. Fast Batch Tokenizer
# =====================================================================

def fast_encode_batch(tokenizer: Any, texts: List[str]) -> List[List[int]]:
    if not texts:
        return []
    enc = getattr(tokenizer, "enc", None)
    if enc and hasattr(enc, "encode_batch"):
        try:
            return enc.encode_batch(texts, num_threads=16, allowed_special="all")
        except Exception:
            pass
    return [tokenizer.encode(t) for t in texts]


# =====================================================================
# 3. Stream Readers with Schema Safeguards
# =====================================================================

class SourceReader:
    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight

    def stream_docs(self, tokenizer) -> Iterator[List[int]]:
        raise NotImplementedError


class FastParquetReader(SourceReader):
    """Streams tokenized documents with fail-fast schema verification."""
    def __init__(
        self,
        name: str,
        repo: str,
        subset: Optional[str],
        text_key: str,
        weight: float,
        data_dir: Optional[str] = None,
    ):
        super().__init__(name, weight)
        self.repo = repo
        self.subset = subset
        self.text_key = text_key
        self.data_dir = data_dir
        self._target_files: Optional[List[str]] = None

    def _resolve_target_files(self, token: Optional[str]) -> List[str]:
        api = HfApi(token=token)
        try:
            files = api.list_repo_files(repo_id=self.repo, repo_type="dataset")
            valid = [f for f in files if f.endswith((".parquet", ".jsonl", ".jsonl.gz"))]
            if self.data_dir:
                valid = [f for f in valid if f.startswith(self.data_dir.strip("/"))]
            if self.subset:
                valid = [f for f in valid if self.subset in f]
            parquets = [f for f in valid if f.endswith(".parquet")]
            candidates = parquets if parquets else valid
            candidates.sort()
            return candidates
        except Exception as e:
            print(f"[{self.name}] Error checking repo {self.repo}: {e}")
            return []

    def _stream_from_parquet(self, path: str, tokenizer) -> Iterator[List[int]]:
        pf = pq.ParquetFile(path)
        schema_cols = pf.schema_arrow.names

        col_to_use = self.text_key if self.text_key in schema_cols else None
        if not col_to_use:
            for candidate in ["text", "content", "code", "solution"]:
                if candidate in schema_cols:
                    col_to_use = candidate
                    break

        if not col_to_use:
            raise KeyError(
                f"[{self.name}] FATAL: No text column found in {path}! "
                f"Available: {schema_cols}. Refusing to fall back to metadata."
            )

        for batch in pf.iter_batches(batch_size=2048, columns=[col_to_use]):
            raw_texts = batch[col_to_use].to_pylist()
            valid_texts = [t for t in raw_texts if t and isinstance(t, str) and len(t.strip()) > 30]
            if not valid_texts:
                continue
            for doc in fast_encode_batch(tokenizer, valid_texts):
                if doc:
                    yield doc

    def stream_docs(self, tokenizer) -> Iterator[List[int]]:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if self._target_files is None:
            self._target_files = self._resolve_target_files(token)
            if not self._target_files:
                raise FileNotFoundError(f"[{self.name}] No valid data files resolved in {self.repo}")
            print(f"[{self.name}] Resolved {len(self._target_files)} shard(s) from {self.repo}")

        for target_file in self._target_files:
            try:
                local_path = hf_hub_download(
                    repo_id=self.repo,
                    filename=target_file,
                    repo_type="dataset",
                    token=token,
                )
            except Exception as e:
                print(f"[{self.name}] Download failed for {target_file}: {e}")
                continue

            try:
                yield from self._stream_from_parquet(local_path, tokenizer)
            except Exception as e:
                print(f"[{self.name}] Error processing {local_path}: {e}")


class FilteredRepoCodeReader(FastParquetReader):
    """Streams Python source files from The Stack, filtering boilerplate in-flight."""
    FORBIDDEN_KEYWORDS = [
        "django.db", "rest_framework", "boto3", "flask", "conftest",
        "alembic", "Auto-generated", "protobuf", "setup(", "argparse"
    ]

    def _stream_from_parquet(self, path: str, tokenizer) -> Iterator[List[int]]:
        pf = pq.ParquetFile(path)
        schema_cols = pf.schema_arrow.names

        if self.text_key not in schema_cols:
            raise KeyError(f"[{self.name}] Missing '{self.text_key}' in {schema_cols}")

        for batch in pf.iter_batches(batch_size=2048, columns=[self.text_key]):
            raw_files = batch[self.text_key].to_pylist()
            clean_texts = []
            for code in raw_files:
                if not code or not isinstance(code, str):
                    continue
                if len(code) < 200 or len(code) > 25000:
                    continue
                if "def " not in code:
                    continue
                if not any(k in code for k in ["for ", "while ", "if "]):
                    continue
                if any(bad in code for bad in self.FORBIDDEN_KEYWORDS):
                    continue
                clean_texts.append(code.strip())

            if clean_texts:
                for doc in fast_encode_batch(tokenizer, clean_texts):
                    if doc:
                        yield doc


class JSONLStreamReader(SourceReader):
    """Streams JSONL/JSONL.gz datasets (e.g., tohio/slm-synthetic-pretrain)."""
    def __init__(self, name: str, repo: str, weight: float, text_key: str = "text"):
        super().__init__(name, weight)
        self.repo = repo
        self.text_key = text_key
        self._target_files: Optional[List[str]] = None

    def _resolve_files(self, token: Optional[str]) -> List[str]:
        api = HfApi(token=token)
        files = api.list_repo_files(repo_id=self.repo, repo_type="dataset")
        return [f for f in files if f.endswith((".jsonl", ".jsonl.gz"))]

    def stream_docs(self, tokenizer) -> Iterator[List[int]]:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if self._target_files is None:
            self._target_files = self._resolve_files(token)
            print(f"[{self.name}] Resolved {len(self._target_files)} JSONL file(s)")

        for f in self._target_files:
            local_path = hf_hub_download(repo_id=self.repo, filename=f, repo_type="dataset", token=token)
            opener = gzip.open if local_path.endswith(".gz") else open
            buffer = []

            with opener(local_path, "rt", encoding="utf-8") as file_in:
                for line in file_in:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        text = record.get(self.text_key, "").strip()
                        if text:
                            buffer.append(text)
                    except Exception:
                        continue

                    if len(buffer) >= 2048:
                        for doc in fast_encode_batch(tokenizer, buffer):
                            if doc:
                                yield doc
                        buffer = []

            if buffer:
                for doc in fast_encode_batch(tokenizer, buffer):
                    if doc:
                        yield doc


class ReasoningParquetReader(FastParquetReader):
    """Stitches mathematical problems and reasoning solutions into formatted text."""
    def __init__(self, name: str, repo: str, problem_key: str, solution_key: str, weight: float, wrap_think: bool = False):
        super().__init__(name=name, repo=repo, subset=None, text_key=problem_key, weight=weight)
        self.problem_key = problem_key
        self.solution_key = solution_key
        self.wrap_think = wrap_think

    def _stream_from_parquet(self, path: str, tokenizer) -> Iterator[List[int]]:
        pf = pq.ParquetFile(path)
        schema_cols = pf.schema_arrow.names

        p_col = self.problem_key if self.problem_key in schema_cols else "problem"
        s_col = self.solution_key if self.solution_key in schema_cols else "solution"

        if p_col not in schema_cols or s_col not in schema_cols:
            raise KeyError(f"[{self.name}] Missing keys in {schema_cols}")

        for batch in pf.iter_batches(batch_size=2048, columns=[p_col, s_col]):
            probs = batch[p_col].to_pylist()
            sols = batch[s_col].to_pylist()
            docs = []
            for p, s in zip(probs, sols):
                if not (p and s and isinstance(p, str) and isinstance(s, str)):
                    continue
                p_c, s_c = p.strip(), s.strip()
                if not p_c or not s_c:
                    continue
                sol_fmt = f"<think>\n{s_c}\n</think>" if (self.wrap_think and "<think>" not in s_c) else s_c
                docs.append(f"Problem:\n{p_c}\n\nSolution:\n{sol_fmt}")

            if docs:
                for doc in fast_encode_batch(tokenizer, docs):
                    if doc:
                        yield doc


# =====================================================================
# 4. Physical Packing Engine
# =====================================================================

def pack_curriculum_to_shards(
    sources: List[SourceReader],
    output_dir: str = "data/pretrain",
    total_token_budget: int = 10_000_000_000,
    shard_size_tokens: int = 25_000_000,
    val_tokens: Optional[int] = None,
    tokenizer_type: str = "tiktoken",
    tokenizer_path: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = get_tokenizer(tokenizer_type, tokenizer_path)
    eot_id = tokenizer.eot_id

    if val_tokens is None:
        val_tokens = max(50_000, min(2_000_000, total_token_budget // 200))

    raw_weights = [s.weight for s in sources]
    total_w = sum(raw_weights)
    target_proportions = [w / total_w for w in raw_weights]
    num_sources = len(sources)

    print("=" * 80)
    print("      Unified Physical Token Packing Engine (125M / 350M / 1B)")
    print("=" * 80)
    print(f"Total Target Budget: {total_token_budget:,} tokens")
    print(f"Validation Target:   {val_tokens:,} tokens")
    print(f"Shard Size:          {shard_size_tokens:,} tokens")
    print(f"Output Directory:    {output_dir}")
    print("Curriculum Allocation:")
    for s, p in zip(sources, target_proportions):
        print(f"  • {s.name:<25} {p * 100:>5.1f}% ({int(p * total_token_budget):,} tokens)")
    print("=" * 80)

    generators = [s.stream_docs(tokenizer) for s in sources]
    tokens_per_source = [0] * num_sources

    # 1. Harvest validation partition
    print("\nHarvesting validation partition...")
    val_buffer = []
    src_idx = 0
    while len(val_buffer) < val_tokens:
        try:
            doc = next(generators[src_idx % num_sources])
            val_buffer.extend(doc)
            val_buffer.append(eot_id)
        except StopIteration:
            generators[src_idx % num_sources] = sources[src_idx % num_sources].stream_docs(tokenizer)
        src_idx += 1

    val_data = np.array(val_buffer[:val_tokens], dtype=np.uint16)
    val_file = os.path.join(output_dir, "val_00000.bin")
    val_data.tofile(val_file)
    print(f"✓ Validation shard complete: {val_file} ({len(val_data):,} tokens)")

    # 2. Deficit-based training harvest
    print("\nPacking training shards...")
    token_buffer = []
    shard_idx = 0
    total_written = 0

    while total_written + len(token_buffer) < total_token_budget:
        total_so_far = max(1, sum(tokens_per_source))
        best_i = 0
        max_deficit = -1e9
        for i in range(num_sources):
            defic = target_proportions[i] - (tokens_per_source[i] / total_so_far)
            if defic > max_deficit:
                max_deficit = defic
                best_i = i

        chosen = best_i
        try:
            doc = next(generators[chosen])
        except StopIteration:
            generators[chosen] = sources[chosen].stream_docs(tokenizer)
            try:
                doc = next(generators[chosen])
            except StopIteration:
                continue

        token_buffer.extend(doc)
        token_buffer.append(eot_id)
        tokens_per_source[chosen] += len(doc) + 1

        while len(token_buffer) >= shard_size_tokens:
            shard_data = np.array(token_buffer[:shard_size_tokens], dtype=np.uint16)
            out_file = os.path.join(output_dir, f"train_{shard_idx:05d}.bin")
            shard_data.tofile(out_file)

            total_written += shard_size_tokens
            token_buffer = token_buffer[shard_size_tokens:]
            pct = (total_written / total_token_budget) * 100
            print(f"✓ Wrote {out_file} | Total: {total_written:,} / {total_token_budget:,} ({pct:.1f}%)")
            shard_idx += 1

            if total_written >= total_token_budget:
                break

    if token_buffer and total_written < total_token_budget:
        shard_data = np.array(token_buffer, dtype=np.uint16)
        out_file = os.path.join(output_dir, f"train_{shard_idx:05d}.bin")
        shard_data.tofile(out_file)
        total_written += len(shard_data)
        print(f"✓ Final shard written: {out_file} ({len(shard_data):,} tokens)")

    print(f"\nAll shards generated. Total training tokens: {total_written:,}")


def build_curriculum(total_tokens: int, param_count: int) -> List[SourceReader]:
    w = compute_curriculum_weights(total_tokens=total_tokens, param_count=param_count)
    return [
        FastParquetReader("FineWeb-Edu", "HuggingFaceFW/fineweb-edu", "sample/100BT", "text", weight=w["FineWeb-Edu"]),
        FastParquetReader("DCLM-Edu", "HuggingFaceTB/dclm-edu", None, "text", weight=w["DCLM-Edu"]),
        FilteredRepoCodeReader("The-Stack-Filtered", "bigcode/the-stack-dedup", None, "content", weight=w["The-Stack-Filtered"], data_dir="data/python"),
        ReasoningParquetReader("OpenMathReasoning", "nvidia/OpenMathReasoning", "problem", "generated_solution", weight=w["OpenMathReasoning"]),
        ReasoningParquetReader("NuminaMath-CoT", "AI-MO/NuminaMath-CoT", "problem", "solution", weight=w["NuminaMath-CoT"], wrap_think=True),
        JSONLStreamReader("SLM-Synthetic-Pretrain", "tohio/slm-synthetic-pretrain", weight=w["SLM-Synthetic-Pretrain"], text_key="text"),
    ]


if __name__ == "__main__":
    tokens_target = 10_000_000_000
    param_target = 350_000_000
    shard_size = 25_000_000
    target_out = "data/pretrain"
    val_budget = None

    for arg in sys.argv[1:]:
        if arg.startswith("total_tokens="):
            tokens_target = int(arg.split("=")[1])
        elif arg.startswith("params=") or arg.startswith("param_count="):
            param_target = int(arg.split("=")[1])
        elif arg.startswith("shard_size="):
            shard_size = int(arg.split("=")[1])
        elif arg.startswith("output_dir="):
            target_out = arg.split("=")[1]
        elif arg.startswith("val_tokens="):
            val_budget = int(arg.split("=")[1])

    curriculum = build_curriculum(total_tokens=tokens_target, param_count=param_target)
    pack_curriculum_to_shards(
        sources=curriculum,
        output_dir=target_out,
        total_token_budget=tokens_target,
        shard_size_tokens=shard_size,
        val_tokens=val_budget,
    )