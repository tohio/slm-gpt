"""
data/prepare_dpo.py: Local-caching pairwise preference ingestion engine.

Downloads preference datasets (e.g., argilla/dpo-mix-7k, ultrafeedback)
locally via HF Hub to eliminate streaming connection deadlocks, parses
conversational and direct preference schemas, and writes normalized
prompt/chosen/rejected pairs to JSONL.
"""

import argparse
import gzip
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


def _extract_turn_text(val: Any) -> str:
    """Extracts contiguous text from a raw string or list of turn dictionaries."""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        turns = []
        for turn in val:
            if isinstance(turn, dict):
                content = turn.get("content", "")
                turns.append(str(content).strip())
            else:
                turns.append(str(turn).strip())
        return "\n".join(turns).strip()
    return str(val).strip()


def normalize_preference_record(record: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Normalizes multi-turn message lists, conversational transcripts,
    and flat prompt/chosen/rejected records into standard DPO schema.
    """
    prompt = record.get("prompt", "")
    chosen = record.get("chosen", "")
    rejected = record.get("rejected", "")
    system = record.get("system", "")

    # Multi-turn conversation format (e.g., argilla/dpo-mix-7k, ultrafeedback)
    if isinstance(chosen, list) and len(chosen) > 0 and isinstance(chosen[0], dict):
        if not prompt and len(chosen) >= 2:
            prompt_turns = [t.get("content", "") for t in chosen[:-1] if t.get("role") != "system"]
            sys_turns = [t.get("content", "") for t in chosen[:-1] if t.get("role") == "system"]
            if sys_turns and not system:
                system = "\n".join(sys_turns)
            prompt = "\n".join(prompt_turns)
            chosen = chosen[-1].get("content", "")
        elif len(chosen) == 1:
            chosen = chosen[0].get("content", "")
        else:
            chosen = _extract_turn_text(chosen)

    if isinstance(rejected, list) and len(rejected) > 0 and isinstance(rejected[0], dict):
        if len(rejected) >= 2 and not prompt:
            rejected = rejected[-1].get("content", "")
        elif len(rejected) == 1:
            rejected = rejected[0].get("content", "")
        else:
            rejected = _extract_turn_text(rejected)

    prompt_str = _extract_turn_text(prompt)
    chosen_str = _extract_turn_text(chosen)
    rejected_str = _extract_turn_text(rejected)
    system_str = _extract_turn_text(system) if system else ""

    if not prompt_str or not chosen_str or not rejected_str:
        return None

    # Filter identical completions
    if chosen_str == rejected_str:
        return None

    out = {
        "prompt": prompt_str,
        "chosen": chosen_str,
        "rejected": rejected_str,
    }
    if system_str:
        out["system"] = system_str
    return out


def _stream_from_local_file(path: str) -> Iterator[Dict[str, Any]]:
    """Yields parsed dictionaries from local parquet, jsonl, or jsonl.gz files."""
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
    """Finds all parquet or jsonl shards in a Hugging Face dataset repo."""
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
        print(f"[DPO Loader] Error querying repository '{repo}': {e}")
        return []


def stream_preference_source(
    source: str,
    subset: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """Yields raw preference items from local files or downloaded HF shards."""
    source_path = Path(source)
    if source_path.exists():
        print(f"[DPO Loader] Ingesting local file: {source}")
        yield from _stream_from_local_file(str(source_path))
        return

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    print(f"[DPO Loader] Resolving remote shards for '{source}'...")
    shards = _resolve_hf_shards(source, subset, token)

    if not shards:
        raise FileNotFoundError(f"No valid .parquet or .jsonl data files found in '{source}'.")

    for shard in shards:
        print(f"[DPO Loader] Downloading shard: {shard}")
        local_path = hf_hub_download(
            repo_id=source,
            filename=shard,
            repo_type="dataset",
            token=token,
        )
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"[DPO Loader] Cached locally: {os.path.basename(local_path)} ({file_size_mb:.1f} MB)")
        yield from _stream_from_local_file(local_path)


def prepare_dpo_dataset(
    dataset_name: str,
    output_path: str = "data/dpo/preference_pairs.jsonl",
    subset: Optional[str] = None,
    max_samples: Optional[int] = 7_000,
):
    """Downloads, standardizes, and writes preference pairs to local JSONL."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    written_count = 0
    pbar = tqdm(total=max_samples, unit="pairs", desc="Processing DPO Pairs")

    with open(out_file, "w", encoding="utf-8") as f_out:
        for raw_record in stream_preference_source(dataset_name, subset=subset):
            record = normalize_preference_record(raw_record)
            if not record:
                continue

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written_count += 1
            pbar.update(1)

            if max_samples and written_count >= max_samples:
                break

    pbar.close()
    file_size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"\n✓ DPO Preparation Complete:")
    print(f"  -> File: {output_path} ({written_count:,} pairs, {file_size_mb:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare pairwise DPO preferences.")
    parser.add_argument("--dataset_name", type=str, default="argilla/dpo-mix-7k", help="Hugging Face repo or local path")
    parser.add_argument("--output_path", type=str, default="data/dpo/preference_pairs.jsonl", help="Destination JSONL path")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset/config")
    parser.add_argument("--max_samples", type=int, default=7000, help="Maximum preference pairs to collect")

    args = parser.parse_args()
    prepare_dpo_dataset(
        dataset_name=args.dataset_name,
        output_path=args.output_path,
        subset=args.subset,
        max_samples=args.max_samples,
    )