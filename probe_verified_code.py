import os
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

TESTS = [
    {
        "name": "Cosmopedia-v2 (python-edu)",
        "repo": "HuggingFaceTB/cosmopedia-v2",
        "kwargs": {"name": "python-edu", "split": "train", "streaming": True, "token": token},
    },
    {
        "name": "Cosmopedia-v1 (python_code)",
        "repo": "HuggingFaceTB/cosmopedia",
        "kwargs": {"name": "python_code", "split": "train", "streaming": True, "token": token},
    },
    {
        "name": "The-Stack-Dedup (Python via data_dir)",
        "repo": "bigcode/the-stack-dedup",
        "kwargs": {"data_dir": "data/python", "split": "train", "streaming": True, "token": token},
    },
    {
        "name": "Magicoder-OSS (Filtered to Python)",
        "repo": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "kwargs": {"split": "train", "streaming": True, "token": token},
        "filter_python": True,
    },
    {
        "name": "Tiny-Codes (High-Density Educational)",
        "repo": "nampdn-ai/tiny-codes",
        "kwargs": {"split": "train", "streaming": True, "token": token},
    },
]

print("=" * 80)
print("VERIFYING EXACT PYTHON DATASET SCHEMAS & CONTENT")
print("=" * 80)

for test in TESTS:
    print(f"\n>> Testing: {test['name']}")
    try:
        ds = load_dataset(test["repo"], **test["kwargs"])
        
        # If filtering for Python specifically
        if test.get("filter_python"):
            sample = None
            for row in ds:
                if str(row.get("lang", "")).lower() == "python":
                    sample = row
                    break
        else:
            sample = next(iter(ds))

        if not sample:
            print("   ✗ No sample retrieved.")
            continue

        cols = list(sample.keys())
        print(f"   Columns: {cols}")

        # Check for actual code text content
        target_col = None
        for cand in ["text", "content", "code", "solution", "response"]:
            if cand in sample and isinstance(sample[cand], str) and len(sample[cand].strip()) > 0:
                target_col = cand
                break

        if target_col:
            val = sample[target_col].strip()
            # Verify it's not a hash/blob ID
            if len(val) == 40 and not any(k in val for k in ["def ", "import ", "class ", "="]):
                print(f"   ✗ WARNING: Column '{target_col}' appears to be a raw hash/ID! Value: {val}")
            else:
                print(f"   ✓ VALID CODE TEXT found in '{target_col}' ({len(val):,} chars)")
                preview = "\n".join("      " + line for line in val[:300].splitlines())
                print(f"   --- Preview ---\n{preview}\n   ---------------")
        else:
            # Print raw values to see what is inside
            short_sample = {k: (str(v)[:60] + "..." if len(str(v)) > 60 else v) for k, v in sample.items()}
            print(f"   ✗ No text column found. Raw row: {short_sample}")

    except Exception as e:
        print(f"   ✗ Error: {e}")

print("\n" + "=" * 80)
