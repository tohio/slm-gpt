"""
data/prepare_dpo.py: Preference Pair Harvester & Normalizer for DPO.
Normalizes conversational lists and QA pairs, filters identical answers
and verbosity exploits, and persists preference pairs to JSONL.
Supports both key=value CLI invocations and standard flag arguments.
"""

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def normalize_preference_record(sample: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Normalizes raw preference data into canonical {system, prompt, chosen, rejected}.

    Returns None if:
      - Formats cannot be extracted.
      - Chosen and rejected are identical.
      - Length exceeds the 350-word verbosity threshold or extreme length disparity.
    """
    system_text = sample.get("system", "")
    prompt_text = ""
    chosen_text = ""
    rejected_text = ""

    # 1. Conversational format: list of role/content dicts
    if isinstance(sample.get("chosen"), list) and isinstance(sample.get("rejected"), list):
        chosen_list = sample["chosen"]
        rejected_list = sample["rejected"]

        # Extract system prompt if present
        for msg in chosen_list:
            if isinstance(msg, dict) and msg.get("role") == "system":
                system_text = msg.get("content", "").strip()
                break

        # Extract user prompt (last user turn)
        for msg in reversed(chosen_list):
            if isinstance(msg, dict) and msg.get("role") == "user":
                prompt_text = msg.get("content", "").strip()
                break

        # Extract assistant responses
        if chosen_list and isinstance(chosen_list[-1], dict) and chosen_list[-1].get("role") == "assistant":
            chosen_text = chosen_list[-1].get("content", "").strip()
        if rejected_list and isinstance(rejected_list[-1], dict) and rejected_list[-1].get("role") == "assistant":
            rejected_text = rejected_list[-1].get("content", "").strip()

    # 2. Flat QA format
    else:
        prompt_text = str(sample.get("prompt", "") or sample.get("question", "")).strip()
        chosen_text = str(sample.get("chosen", "")).strip()
        rejected_text = str(sample.get("rejected", "")).strip()

    # Integrity gates
    if not prompt_text or not chosen_text or not rejected_text:
        return None

    # Filter identical completions
    if chosen_text == rejected_text:
        return None

    # Filter verbosity hacks & length exploits (>350 words or extreme length imbalance)
    chosen_words = len(chosen_text.split())
    rejected_words = len(rejected_text.split())

    if chosen_words > 350 or rejected_words > 350:
        return None

    max_len = max(chosen_words, rejected_words)
    min_len = max(1, min(chosen_words, rejected_words))
    if (max_len / min_len > 8.0) and (max_len > 100):
        return None

    record = {
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }
    if system_text:
        record["system"] = system_text

    return record


def prepare_dpo_dataset(
    output_path: str = "data/dpo/preference_pairs.jsonl",
    dataset_name: str = "argilla/dpo-mix-7k",
    total_samples: int = 7_000,
):
    """Harvests and normalizes preference pairs from Hugging Face or local cache."""
    print("=" * 70)
    print("      slm-gpt DPO Preference Pair Assembler (Offline Engine)       ")
    print("=" * 70)
    print(f"Target Budget: {total_samples:,} pairs")
    print(f"Source:        {dataset_name}")
    print(f"Output:        {output_path}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    from datasets import load_dataset

    print(f"Loading '{dataset_name}' from cache/Hub...")
    ds = load_dataset(dataset_name, split="train")

    valid_records: List[Dict[str, str]] = []
    skipped = 0

    for row in ds:
        norm = normalize_preference_record(row)
        if norm is not None:
            valid_records.append(norm)
            if len(valid_records) >= total_samples:
                break
        else:
            skipped += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in valid_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("✓ Successfully assembled DPO preference dataset:")
    print(f"  Path:       {output_path}")
    print(f"  Kept:       {len(valid_records):,} pairs")
    print(f"  Filtered:   {skipped:,} pairs (identical / length exploit)")
    print("=" * 70)


def parse_cli_args() -> Dict[str, Any]:
    """Flexible CLI parser supporting both key=value and standard flag arguments."""
    kwargs: Dict[str, Any] = {
        "output_path": "data/dpo/preference_pairs.jsonl",
        "dataset_name": "argilla/dpo-mix-7k",
        "total_samples": 7_000,
    }

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if "=" in arg:
            k, v = arg.split("=", 1)
            k = k.lstrip("-")
            if k in ("total_samples", "max_samples", "budget"):
                kwargs["total_samples"] = int(v)
            elif k in ("output_path", "output_file"):
                kwargs["output_path"] = v
            elif k in ("dataset_name", "source"):
                kwargs["dataset_name"] = v
        elif arg in ("--output_path", "-o"):
            i += 1
            if i < len(args):
                kwargs["output_path"] = args[i]
        elif arg in ("--total_samples", "--max_samples", "-n"):
            i += 1
            if i < len(args):
                kwargs["total_samples"] = int(args[i])
        elif arg in ("--dataset_name", "-d"):
            i += 1
            if i < len(args):
                kwargs["dataset_name"] = args[i]
        elif arg.isdigit():
            kwargs["total_samples"] = int(arg)
        i += 1

    return kwargs


if __name__ == "__main__":
    cli_kwargs = parse_cli_args()
    prepare_dpo_dataset(**cli_kwargs)