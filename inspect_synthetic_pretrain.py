import os
from collections import Counter
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
REPO_ID = "tohio/slm-synthetic-pretrain"

print("=" * 80)
print(f"INSPECTING: {REPO_ID}")
print("=" * 80)

# 1. Inspect Hub metadata and file inventory
api = HfApi(token=token)
try:
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith(".parquet")]
    jsonl_files = [f for f in files if f.endswith((".jsonl", ".jsonl.gz"))]
    print(f"Repository file inventory:")
    print(f"  • Parquet shards: {len(parquet_files)}")
    print(f"  • JSONL shards:   {len(jsonl_files)}")
    print(f"  • Total files:    {len(files)}")
except Exception as e:
    print(f"Could not list repo files: {e}")

# 2. Stream and profile samples
try:
    ds = load_dataset(REPO_ID, split="train", streaming=True, token=token)
    
    first_row = next(iter(ds))
    cols = list(first_row.keys())
    print(f"\nAvailable Columns / Schema:")
    print(f"  {cols}")

    # Identify primary text column
    text_col = "text" if "text" in cols else ("content" if "content" in cols else cols[0])
    print(f"  Using '{text_col}' as primary content column.")

    print("\n" + "=" * 80)
    print("PROFILING FIRST 20 RECORDS (SPECIAL TOKENS, DENSITY, PAYLOAD)")
    print("=" * 80)

    token_indicators = {
        "<think>": 0,
        "</think>": 0,
        "<tool_call>": 0,
        "<|fim_prefix|>": 0,
        "<|fim_middle|>": 0,
        "<|fim_suffix|>": 0,
        "def ": 0,
        "class ": 0,
        "assert ": 0,
        "Problem:": 0,
        "Solution:": 0,
    }

    lengths = []
    samples_to_print = []

    for idx, row in enumerate(ds.take(20)):
        val = str(row.get(text_col, ""))
        lengths.append(len(val))
        
        for marker in token_indicators:
            if marker in val:
                token_indicators[marker] += 1

        if idx < 3:
            samples_to_print.append((idx + 1, len(val), val))

    avg_len = sum(lengths) / max(1, len(lengths))
    print(f"Sample length stats (first 20 rows):")
    print(f"  • Min length: {min(lengths):,} chars")
    print(f"  • Max length: {max(lengths):,} chars")
    print(f"  • Avg length: {avg_len:,.1f} chars (~{int(avg_len / 4):,} tokens/doc)")

    print("\nMarker / Feature Frequency (out of 20 samples):")
    for marker, count in token_indicators.items():
        if count > 0:
            print(f"  • {marker:<18} : {count}/20 ({count/20*100:.0f}%)")

    for num, length, content in samples_to_print:
        print(f"\n{'-'*80}")
        print(f"SAMPLE {num} | Length: {length:,} chars")
        print(f"{'-'*80}")
        preview_lines = content.strip().splitlines()
        print("\n".join(preview_lines[:25]))
        if len(preview_lines) > 25:
            print(f"... [{len(preview_lines) - 25} lines truncated] ...")

except Exception as e:
    print(f"Failed to stream dataset: {e}")

print("\n" + "=" * 80)
