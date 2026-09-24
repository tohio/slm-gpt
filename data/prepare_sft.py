"""
data/prepare_sft_dataset.py: Curated 4-Pillar SFT Dataset Generator (Local Disk Ingestion).
Downloads parquet shards locally via hf_hub_download to eliminate HTTP streaming deadlocks,
extracts and sanitizes samples, validates syntax, and writes balanced jsonl files.
"""

import ast
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

random.seed(42)

# =====================================================================
# Sanitation & Validation Helpers
# =====================================================================

PREAMBLE_PATTERNS = [
    r"^(?:Sure|Certainly|Of course|Definitely)[!,.]?\s*(?:I(?:'d| would)? be (?:happy|glad) to (?:help|assist|explain)[^.\n]*[.\n]+)?",
    r"^Here(?:'s| is) (?:the|an?|your) (?:code|solution|implementation|python script|function)[^:\n]*:\s*",
    r"^(?:In this (?:guide|tutorial|script)|To (?:achieve|do|solve) this)[^,\n]*,?\s*",
    r"^(?:Great question|Good question)![ \t]*",
]

POSTAMBLE_PATTERNS = [
    r"\n+(?:I hope (?:this|that) helps[^\n]*|Let me know if you (?:have any (?:other )?questions|need (?:further|more) help)[^\n]*)$",
    r"\n+(?:Feel free to ask if you have any questions|Happy coding!)[^\n]*$",
]

def sanitize_technical_response(text: str) -> str:
    cleaned = text.strip()
    for pat in PREAMBLE_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
    for pat in POSTAMBLE_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned

def extract_python_snippet(text: str) -> Optional[str]:
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if code_blocks:
        return "\n".join(code_blocks)
    return text if ("def " in text or "import " in text or "class " in text) else None

def is_valid_python(code_str: str) -> bool:
    try:
        ast.parse(code_str)
        return True
    except Exception:
        return False

# =====================================================================
# Local Shard Ingestion Helper
# =====================================================================

def download_and_get_batches(repo_id: str, subset_filter: Optional[str], token: Optional[str]):
    """Discovers and downloads parquet files to disk, yielding pyarrow record batches locally."""
    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        parquets = [f for f in files if f.endswith(".parquet")]
        if subset_filter:
            parquets = [f for f in parquets if subset_filter in f]
        parquets.sort()
    except Exception as e:
        print(f"  [Error] Resolving shards for {repo_id}: {e}")
        return

    for filename in parquets:
        try:
            local_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=token,
            )
            pf = pq.ParquetFile(local_file)
            for batch in pf.iter_batches(batch_size=256):
                yield from batch.to_pylist()
        except Exception as e:
            print(f"  [Error] Downloading/reading {filename} from {repo_id}: {e}")
            continue

# =====================================================================
# 4-Pillar Offline Harvest
# =====================================================================

def harvest_code_samples(target_count: int, token: Optional[str]) -> List[Dict[str, Any]]:
    print(f"\n[1/4] Harvesting {target_count:,} Code Execution samples (local cache)...")
    samples = []
    repo = "ise-uiuc/Magicoder-OSS-Instruct-75K"
    
    for row in download_and_get_batches(repo, subset_filter=None, token=token):
        prob = row.get("problem", "").strip()
        sol = row.get("solution", "").strip()
        if not prob or not sol:
            continue

        cleaned_sol = sanitize_technical_response(sol)
        code_cand = extract_python_snippet(cleaned_sol)
        if code_cand and not is_valid_python(code_cand):
            continue

        samples.append({
            "messages": [
                {"role": "user", "content": prob},
                {"role": "assistant", "content": cleaned_sol}
            ],
            "category": "code"
        })
        if len(samples) >= target_count:
            break

    print(f"  ✓ Collected {len(samples):,} code samples.")
    return samples

def harvest_math_reasoning(target_count: int, token: Optional[str]) -> List[Dict[str, Any]]:
    print(f"\n[2/4] Harvesting {target_count:,} Math/Reasoning samples with <think> (local cache)...")
    samples = []
    repo = "open-r1/OpenR1-Math-220k"
    
    for row in download_and_get_batches(repo, subset_filter=None, token=token):
        prob = row.get("problem", "").strip()
        sol = row.get("solution", "").strip()
        if not prob or not sol:
            continue

        if "<think>" in sol and "</think>" in sol:
            formatted_sol = sol
        else:
            formatted_sol = f"<think>\n{sol}\n</think>"

        samples.append({
            "messages": [
                {"role": "user", "content": prob},
                {"role": "assistant", "content": formatted_sol}
            ],
            "category": "math_cot"
        })
        if len(samples) >= target_count:
            break

    print(f"  ✓ Collected {len(samples):,} reasoning samples.")
    return samples

def harvest_constraints(target_count: int, token: Optional[str]) -> List[Dict[str, Any]]:
    print(f"\n[3/4] Harvesting {target_count:,} Constraint Task samples (local cache)...")
    samples = []
    repo = "HuggingFaceTB/smoltalk"

    for row in download_and_get_batches(repo, subset_filter="smol-constraints", token=token):
        msgs = row.get("messages", [])
        if len(msgs) >= 2:
            user_msg = msgs[0]["content"].strip()
            asst_msg = sanitize_technical_response(msgs[1]["content"].strip())
            if user_msg and asst_msg:
                samples.append({
                    "messages": [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": asst_msg}
                    ],
                    "category": "constraint_task"
                })
        if len(samples) >= target_count:
            break

    print(f"  ✓ Collected {len(samples):,} constraint samples.")
    return samples

def harvest_chitchat(target_count: int, token: Optional[str]) -> List[Dict[str, Any]]:
    print(f"\n[4/4] Harvesting {target_count:,} Chit-Chat samples (local cache)...")
    samples = []
    repo = "HuggingFaceTB/smoltalk"

    for row in download_and_get_batches(repo, subset_filter="everyday-conversations", token=token):
        msgs = row.get("messages", [])
        if len(msgs) >= 2:
            cleaned_turns = []
            valid = True
            for m in msgs:
                role = m.get("role", "")
                content = m.get("content", "").strip()
                if not role or not content:
                    valid = False
                    break
                cleaned_turns.append({"role": role, "content": content})
            if valid and len(cleaned_turns) >= 2:
                samples.append({
                    "messages": cleaned_turns,
                    "category": "chitchat"
                })
        if len(samples) >= target_count:
            break

    print(f"  ✓ Collected {len(samples):,} conversational samples.")
    return samples

# =====================================================================
# Main Assembler
# =====================================================================

def prepare_sft_splits(
    total_samples: int = 15_000,
    output_dir: str = "data/sft",
    val_ratio: float = 0.05,
):
    os.makedirs(output_dir, exist_ok=True)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    n_code = int(total_samples * 0.30)
    n_math = int(total_samples * 0.30)
    n_task = int(total_samples * 0.25)
    n_chat = total_samples - (n_code + n_math + n_task)

    print("=" * 70)
    print("   slm-gpt SFT 4-Pillar Dataset Assembler (Local Parquet Engine)")
    print("=" * 70)
    print(f"Target Budget: {total_samples:,} samples")

    dataset: List[Dict[str, Any]] = []
    dataset.extend(harvest_code_samples(n_code, token))
    dataset.extend(harvest_math_reasoning(n_math, token))
    dataset.extend(harvest_constraints(n_task, token))
    dataset.extend(harvest_chitchat(n_chat, token))

    random.shuffle(dataset)

    n_val = max(200, int(len(dataset) * val_ratio))
    val_set = dataset[:n_val]
    train_set = dataset[n_val:]

    train_path = os.path.join(output_dir, "train_sft.jsonl")
    val_path = os.path.join(output_dir, "val_sft.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_set:
            f.write(json.dumps(s) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_set:
            f.write(json.dumps(s) + "\n")

    print("\n" + "=" * 70)
    print("✓ Successfully generated SFT datasets (Offline):")
    print(f"  Train: {train_path} ({len(train_set):,} samples)")
    print(f"  Val:   {val_path} ({len(val_set):,} samples)")
    print("=" * 70)


if __name__ == "__main__":
    budget = 15_000
    if len(sys.argv) > 1:
        budget = int(sys.argv[1].split("=")[-1])
    prepare_sft_splits(total_samples=budget)