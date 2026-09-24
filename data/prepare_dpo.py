"""
data/prepare_dpo.py: Ingest, filter, and format preference datasets for DPO.

Extracts prompt, chosen, and rejected completions into canonical JSONL records:
{
    "system": "Optional system prompt",
    "prompt": "User query",
    "chosen": "High quality / concise response",
    "rejected": "Rambling / low quality / incorrect response"
}
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterator, Optional

from datasets import load_dataset

try:
    from tokenizer.tiktoken_wrap import PretrainedTiktokenTokenizer
except ImportError:
    from tokenizer.tiktoken_tokenizer import PretrainedTiktokenTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare DPO preference pairs for slm-gpt.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="argilla/dpo-mix-7k",
        choices=["argilla/dpo-mix-7k", "Intel/orca_dpo_pairs"],
        help="Hugging Face preference dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/dpo/preference_pairs.jsonl",
        help="Output destination for formatted pairs",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Maximum number of filtered pairs to keep",
    )
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=512,
        help="Maximum allowable tokens in prompt",
    )
    parser.add_argument(
        "--max_response_tokens",
        type=int,
        default=512,
        help="Maximum allowable tokens in chosen/rejected completions",
    )
    parser.add_argument(
        "--min_response_tokens",
        type=int,
        default=4,
        help="Minimum token length to filter degenerate completions",
    )
    return parser.parse_args()


def extract_argilla_record(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Normalizes argilla/dpo-mix-7k conversational format."""
    chosen_conv = row.get("chosen", [])
    rejected_conv = row.get("rejected", [])

    if not chosen_conv or not rejected_conv:
        return None

    # Check for system prompt
    system_prompt = ""
    prompt = ""
    chosen = ""
    rejected = ""

    # Extract user prompt (and optional system prompt)
    for msg in chosen_conv:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        if role == "system":
            system_prompt = content
        elif role == "user":
            prompt = content
        elif role == "assistant":
            chosen = content

    for msg in rejected_conv:
        if msg.get("role") == "assistant":
            rejected = msg.get("content", "").strip()

    if not prompt or not chosen or not rejected:
        return None

    return {
        "system": system_prompt,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def extract_orca_record(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Normalizes Intel/orca_dpo_pairs format."""
    system_prompt = row.get("system", "").strip()
    prompt = row.get("question", "").strip()
    chosen = row.get("chosen", "").strip()
    rejected = row.get("rejected", "").strip()

    if not prompt or not chosen or not rejected:
        return None

    return {
        "system": system_prompt,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def process_dataset(args):
    print(f"--- Preparing DPO Preference Dataset ---")
    print(f"Source: {args.dataset_name} ({args.split} split)")
    print(f"Max Samples: {args.max_samples:,} | Token Caps: prompt<={args.max_prompt_tokens}, resp<={args.max_response_tokens}")

    tokenizer = PretrainedTiktokenTokenizer()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    dataset = load_dataset(args.dataset_name, split=args.split)

    valid_count = 0
    dropped_length = 0
    dropped_identical = 0

    with open(args.output_path, "w", encoding="utf-8") as out_f:
        for row in dataset:
            if valid_count >= args.max_samples:
                break

            if args.dataset_name == "argilla/dpo-mix-7k":
                record = extract_argilla_record(row)
            elif args.dataset_name == "Intel/orca_dpo_pairs":
                record = extract_orca_record(row)
            else:
                record = None

            if not record:
                continue

            # Reject pairs where chosen and rejected are identical
            if record["chosen"] == record["rejected"]:
                dropped_identical += 1
                continue

            # Token length filtering
            p_len = len(tokenizer.encode(record["prompt"]))
            c_len = len(tokenizer.encode(record["chosen"]))
            r_len = len(tokenizer.encode(record["rejected"]))

            if (
                p_len > args.max_prompt_tokens
                or c_len > args.max_response_tokens
                or r_len > args.max_response_tokens
                or c_len < args.min_response_tokens
                or r_len < args.min_response_tokens
            ):
                dropped_length += 1
                continue

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            valid_count += 1

            if valid_count % 500 == 0:
                print(f"  Processed {valid_count:,} valid pairs (dropped: {dropped_length} length, {dropped_identical} identical)...")

    print("=" * 65)
    print(f"✓ Completed DPO pair extraction.")
    print(f"  Saved: {valid_count:,} pairs -> '{args.output_path}'")
    print(f"  Filtered: {dropped_length:,} over-length, {dropped_identical:,} identical")
    print("=" * 65)


if __name__ == "__main__":
    process_dataset(parse_args())