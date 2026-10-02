import os
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

CANDIDATES = [
    {
        "name": "CodeAlpaca-20k",
        "repo": "sahil2801/CodeAlpaca-20k",
        "kwargs": {"split": "train", "streaming": True, "token": token},
        "text_keys": ["output", "instruction"],
    },
    {
        "name": "Evol-Instruct-Code-80k",
        "repo": "nickrosh/Evol-Instruct-Code-80k-v1",
        "kwargs": {"split": "train", "streaming": True, "token": token},
        "text_keys": ["output", "instruction"],
    },
    {
        "name": "Python-Code-23k-ShareGPT",
        "repo": "ajibawa-2023/Python-Code-23k-ShareGPT",
        "kwargs": {"split": "train", "streaming": True, "token": token},
        "text_keys": ["conversations"],
    },
    {
        "name": "CodeFeedback-Filtered-Instruction",
        "repo": "m-a-p/CodeFeedback-Filtered-Instruction",
        "kwargs": {"split": "train", "streaming": True, "token": token},
        "text_keys": ["answer", "query"],
    },
]

print("=" * 80)
print("PROBING HIGH-VOLUME PURE PYTHON ALGORITHMIC DATASETS")
print("=" * 80)

for cand in CANDIDATES:
    print(f"\n>> Testing: {cand['name']} ({cand['repo']})")
    try:
        ds = load_dataset(cand["repo"], **cand["kwargs"])
        sample = next(iter(ds))
        cols = list(sample.keys())
        print(f"   Available Columns: {cols}")

        found_key = None
        for k in cand["text_keys"]:
            if k in sample:
                found_key = k
                break

        if found_key:
            val = sample[found_key]
            if isinstance(val, list):  # ShareGPT format
                val_str = "\n".join(f"{turn.get('from', '')}: {turn.get('value', '')}" for turn in val if isinstance(turn, dict))
            else:
                val_str = str(val)

            print(f"   ✓ PAYLOAD FOUND in '{found_key}' ({len(val_str):,} chars)")
            preview = "\n".join("      " + line for line in val_str.strip()[:350].splitlines())
            print(f"   --- Preview ---\n{preview}\n   ---------------")
        else:
            print(f"   ✗ No candidate key found in columns.")
    except Exception as e:
        print(f"   ✗ Error: {e}")

print("\n" + "=" * 80)
