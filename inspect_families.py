import json
import os
from collections import defaultdict
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
REPO_ID = "tohio/slm-synthetic-pretrain"

print("=" * 80)
print(f"DEEP INSPECTION: {REPO_ID}")
print("=" * 80)

# 1. Print Signal Distribution from Dataset Card / README
print("\n[1] READING DATASET CARD METADATA / README...")
try:
    readme_path = hf_hub_download(REPO_ID, "README.md", repo_type="dataset", token=token)
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()

    if "Signal Distribution" in readme_content:
        dist_section = readme_content.split("Signal Distribution")[1][:1200]
        print("\n--- Signal Distribution Section from README ---")
        print("Signal Distribution" + dist_section.strip())
        print("------------------------------------------------\n")
    else:
        print("No explicit 'Signal Distribution' header found in README. First 600 chars:")
        print(readme_content[:600])
except Exception as e:
    print(f"Could not read README.md: {e}")

# 2. Parse All Records and Group by Signal / Family
print("\n[2] PARSING ALL RECORDS BY SIGNAL / FAMILY...")
ds = load_dataset(REPO_ID, split="train", token=token)

family_stats = defaultdict(lambda: {
    "count": 0,
    "total_chars": 0,
    "samples": [],
    "subtypes": defaultdict(int),
})

for row in ds:
    meta = row.get("metadata", {})
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    elif not isinstance(meta, dict):
        meta = {}

    # Extract signal / family identifier
    signal = (
        meta.get("signal")
        or meta.get("family")
        or meta.get("task")
        or meta.get("type")
        or meta.get("category")
        or "unclassified"
    )

    text = str(row.get("text", "")).strip()
    char_len = len(text)

    stats = family_stats[signal]
    stats["count"] += 1
    stats["total_chars"] += char_len

    # Track any secondary tags or subtypes inside metadata
    for k in ["subtype", "difficulty", "domain", "topic"]:
        if k in meta:
            stats["subtypes"][f"{k}:{meta[k]}"] += 1

    if len(stats["samples"]) < 2:
        stats["samples"].append((char_len, text, meta))

total_records = len(ds)
print(f"Total Records Analyzed: {total_records:,}\n")

# 3. Print Breakdown Table
print(f"{'Signal / Family':<32} | {'Count':<7} | {'Share':<7} | {'Avg Chars':<9} | {'Est. Tokens':<11}")
print("-" * 75)
for sig, data in sorted(family_stats.items(), key=lambda x: x[1]["count"], reverse=True):
    count = data["count"]
    pct = (count / total_records) * 100
    avg_chars = data["total_chars"] / max(1, count)
    est_tokens = int(data["total_chars"] / 3.8)
    print(f"{sig:<32} | {count:<7} | {pct:>5.1f}% | {avg_chars:>9.1f} | {est_tokens:>11,}")

# 4. Print Payloads per Family
print("\n" + "=" * 80)
print("[3] PAYLOAD INSPECTION PER SIGNAL FAMILY")
print("=" * 80)

for sig, data in sorted(family_stats.items(), key=lambda x: x[1]["count"], reverse=True):
    print(f"\n{'#' * 80}")
    print(f"FAMILY: {sig} ({data['count']} records, ~{int(data['total_chars']/3.8):,} tokens)")
    if data["subtypes"]:
        sub_str = ", ".join(f"{k} ({v})" for k, v in data["subtypes"].items())
        print(f"Metadata Tags: {sub_str}")
    print(f"{'#' * 80}")

    for idx, (c_len, text, meta) in enumerate(data["samples"], start=1):
        print(f"\n--- [Sample {idx} | {c_len} chars | Meta: {meta}] ---")
        lines = text.splitlines()
        print("\n".join(lines[:20]))
        if len(lines) > 20:
            print(f"... [{len(lines) - 20} lines truncated] ...")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)
