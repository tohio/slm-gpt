"""
data/prepare_dpo.py: Preference Pair Harvester & Normalizer for DPO.
Harvests multi-turn conversational preferences from allenai/llama-3.1-tulu-3-8b-preference-mixture
and tohio/slm-synthetic-dpo, filters identical completions and length/verbosity exploits,
authenticates via HF_TOKEN, and persists standardized preference pairs to JSONL.
"""

import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path regardless of execution context
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

TULU_DATASET = "allenai/llama-3.1-tulu-3-8b-preference-mixture"
SYNTHETIC_DATASET = "tohio/slm-synthetic-dpo"


def normalize_preference_record(sample: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Normalizes raw preference data into canonical {prompt, chosen, rejected, [system]}.

    Returns None if:
      - Formats cannot be extracted.
      - Chosen and rejected completions are identical.
      - Completion length exceeds 350 words or exhibits an extreme verbosity ratio (>8x).
    """
    system_text = sample.get("system", "")
    prompt_text = ""
    chosen_text = ""
    rejected_text = ""

    # Top-level prompt string if present
    if sample.get("prompt") and isinstance(sample["prompt"], str):
        prompt_text = sample["prompt"].strip()

    # 1. Conversational format: list of message dictionaries
    if isinstance(sample.get("chosen"), list) and isinstance(sample.get("rejected"), list):
        chosen_list = sample["chosen"]
        rejected_list = sample["rejected"]

        # Extract system prompt if present and not already set
        if not system_text:
            for msg in chosen_list:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    system_text = msg.get("content", "").strip()
                    break

        # Extract user prompt from last user turn if not present at root
        if not prompt_text:
            for msg in reversed(chosen_list):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    prompt_text = msg.get("content", "").strip()
                    break

        # Extract assistant completions
        if chosen_list and isinstance(chosen_list[-1], dict) and chosen_list[-1].get("role") == "assistant":
            chosen_text = chosen_list[-1].get("content", "").strip()
        if rejected_list and isinstance(rejected_list[-1], dict) and rejected_list[-1].get("role") == "assistant":
            rejected_text = rejected_list[-1].get("content", "").strip()

    # 2. Flat string completion fallback
    else:
        if not prompt_text:
            prompt_text = str(
                sample.get("prompt", "")
                or sample.get("question", "")
                or sample.get("instruction", "")
            ).strip()
        chosen_text = str(sample.get("chosen", "")).strip()
        rejected_text = str(sample.get("rejected", "")).strip()

    # Integrity gate: All three core fields must be populated
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


def harvest_dataset_records(dataset_name: str, budget: int) -> Tuple[List[Dict[str, str]], int]:
    """Streams and normalizes up to `budget` valid records from a Hugging Face dataset."""
    from datasets import load_dataset

    print(f"Loading '{dataset_name}' (target: {budget:,} pairs)...")
    valid_records: List[Dict[str, str]] = []
    skipped = 0

    try:
        ds = load_dataset(dataset_name, split="train", token=HF_TOKEN, streaming=True)
        for row in ds:
            norm = normalize_preference_record(row)
            if norm is not None:
                valid_records.append(norm)
                if len(valid_records) >= budget:
                    break
            else:
                skipped += 1
    except Exception as e:
        print(f"  ⚠️ Warning loading {dataset_name}: {e}")

    print(f"  ✓ Collected {len(valid_records):,} pairs from {dataset_name} ({skipped:,} filtered)")
    return valid_records, skipped


def prepare_dpo_dataset(
    output_path: str = "data/dpo/preference_pairs.jsonl",
    total_samples: int = 7_000,
    synthetic_ratio: float = 0.30,
    dataset_name: Optional[str] = None,
):
    """Harvests, mixes, and normalizes preference pairs into JSONL."""
    print("=" * 70)
    print("      slm-gpt DPO Preference Pair Assembler (Offline Engine)       ")
    print("=" * 70)
    print(f"Target Budget:   {total_samples:,} pairs")
    print(f"Output:          {output_path}")
    print(f"Auth Token:      {'✓ Detected' if HF_TOKEN else '⚠️ Not Found (Falling back to unauthenticated)'}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    valid_records: List[Dict[str, str]] = []
    total_skipped = 0

    # If user explicitly overrode with a single dataset name, harvest solely from that source
    if dataset_name:
        records, skipped = harvest_dataset_records(dataset_name, total_samples)
        valid_records.extend(records)
        total_skipped += skipped
    else:
        # Default dual-mix: (1 - synthetic_ratio) Tulu-3 + (synthetic_ratio) slm-synthetic-dpo
        budget_synthetic = int(total_samples * synthetic_ratio)
        budget_tulu = total_samples - budget_synthetic

        print(f"Strategy:        Composite ({100 - int(synthetic_ratio * 100)}% Tulu / {int(synthetic_ratio * 100)}% Synthetic)")
        print("-" * 70)

        tulu_records, tulu_skipped = harvest_dataset_records(TULU_DATASET, budget_tulu)
        valid_records.extend(tulu_records)
        total_skipped += tulu_skipped

        synth_records, synth_skipped = harvest_dataset_records(SYNTHETIC_DATASET, budget_synthetic)
        valid_records.extend(synth_records)
        total_skipped += synth_skipped

    # Shuffle composite mixture
    random.seed(42)
    random.shuffle(valid_records)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in valid_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("✓ Successfully assembled DPO preference dataset:")
    print(f"  Path:       {output_path}")
    print(f"  Kept:       {len(valid_records):,} pairs")
    print(f"  Filtered:   {total_skipped:,} pairs (identical / length exploit)")
    print("=" * 70)


def parse_cli_args() -> Dict[str, Any]:
    """Flexible CLI parser supporting both key=value and standard flag arguments."""
    kwargs: Dict[str, Any] = {
        "output_path": "data/dpo/preference_pairs.jsonl",
        "total_samples": 7_000,
        "synthetic_ratio": 0.30,
        "dataset_name": None,
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
            elif k in ("synthetic_ratio", "ratio"):
                kwargs["synthetic_ratio"] = float(v)
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
        elif arg in ("--synthetic_ratio", "-r"):
            i += 1
            if i < len(args):
                kwargs["synthetic_ratio"] = float(args[i])
        elif arg.isdigit():
            kwargs["total_samples"] = int(arg)
        i += 1

    return kwargs


if __name__ == "__main__":
    cli_kwargs = parse_cli_args()
    prepare_dpo_dataset(**cli_kwargs)