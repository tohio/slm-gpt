"""
data/prepare_dpo.py: Local-caching pairwise preference ingestion engine.

Downloads preference datasets (default: allenai/llama-3.1-tulu-3-8b-preference-mixture)
locally via HF Hub, samples uniformly across shards to guarantee multi-domain coverage,
filters identical and length-exploited pairs, and writes normalized JSONL.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
from tqdm import tqdm

random.seed(42)


def _extract_turn_text(val: Any) -> str:
    """Extracts raw text from either a string or a list of message dicts."""
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
    """Standardizes records into raw prompt/chosen/rejected text."""
    prompt = record.get("prompt", "")
    chosen = record.get("chosen", "")
    rejected = record.get("rejected", "")
    system = record.get("system", "")

    # Multi-turn conversation format
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

    # Guardrail against verbosity reward hacking
    len_c = len(chosen_str.split())
    len_r = len(rejected_str.split())
    if len_c > 2.2 * max(len_r, 1) and len_c > 350:
        return None

    out = {
        "prompt": prompt_str,
        "chosen": chosen_str,
        "rejected": rejected_str,
    }
    if system_str:
        out["system"] = system_str
    return out


def resolve_shards(repo_id: str, subset: Optional[str], token: Optional[str]) -> List[str]:
    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        parquets = [f for f in files if f.endswith(".parquet")]
        if subset:
            parquets = [f for f in parquets if subset in f]
        return sorted(parquets)
    except Exception as e:
        print(f"[Error] Failed to resolve shards in '{repo_id}': {e}")
        return []


def prepare_dpo_dataset(
    dataset_name: str = "allenai/llama-3.1-tulu-3-8b-preference-mixture",
    output_path: str = "data/dpo/preference_pairs.jsonl",
    subset: Optional[str] = None,
    max_samples: int = 7000,
):
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    print(f"[DPO Loader] Source: {dataset_name} | Target: {max_samples:,} pairs")

    local_source = Path(dataset_name)
    if local_source.exists():
        shards = sorted([str(p) for p in local_source.glob("*.parquet")]) if local_source.is_dir() else [str(local_source)]
        is_local = True
    else:
        shards = resolve_shards(dataset_name, subset, token)
        is_local = False

    if not shards:
        raise FileNotFoundError(f"No parquet shards found for '{dataset_name}'.")

    quota_per_shard = max(1, math.ceil(max_samples / len(shards)))
    collected_pairs: List[Dict[str, str]] = []

    for i, shard in enumerate(shards):
        if is_local:
            local_path = shard
        else:
            print(f"[{i + 1}/{len(shards)}] Downloading remote shard: {shard}")
            local_path = hf_hub_download(
                repo_id=dataset_name,
                filename=shard,
                repo_type="dataset",
                token=token,
            )

        shard_pairs = []
        pf = pq.ParquetFile(local_path)
        for batch in pf.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                norm = normalize_preference_record(row)
                if norm:
                    shard_pairs.append(norm)
                if len(shard_pairs) >= quota_per_shard:
                    break
            if len(shard_pairs) >= quota_per_shard:
                break

        collected_pairs.extend(shard_pairs)
        print(f"  ✓ Harvested {len(shard_pairs):,} pairs from shard {i + 1} (Total: {len(collected_pairs):,})")
        if len(collected_pairs) >= max_samples:
            break

    random.shuffle(collected_pairs)
    final_pairs = collected_pairs[:max_samples]

    with open(out_file, "w", encoding="utf-8") as f_out:
        for item in final_pairs:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    file_size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"\n✓ DPO Preparation Complete: {output_path} ({len(final_pairs):,} pairs, {file_size_mb:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare pairwise DPO preferences.")
    parser.add_argument("--dataset_name", type=str, default="allenai/llama-3.1-tulu-3-8b-preference-mixture")
    parser.add_argument("--output_path", type=str, default="data/dpo/preference_pairs.jsonl")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=7000)

    args = parser.parse_args()
    prepare_dpo_dataset(
        dataset_name=args.dataset_name,
        output_path=args.output_path,
        subset=args.subset,
        max_samples=args.max_samples,
    )