from datasets import load_dataset

CANDIDATES = [
    {
        "name": "Cosmopedia-v2 (python_code)",
        "repo": "HuggingFaceTB/cosmopedia-v2",
        "subset": "python_code",
        "split": "train",
        "candidate_keys": ["text", "code", "content"],
    },
    {
        "name": "Magicoder-OSS-Instruct-75K",
        "repo": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "subset": None,
        "split": "train",
        "candidate_keys": ["solution", "problem", "code"],
    },
    {
        "name": "The-Stack-Dedup (Python)",
        "repo": "bigcode/the-stack-dedup",
        "subset": "data/python",
        "split": "train",
        "candidate_keys": ["content"],
    },
]

print("=" * 80)
print("PROBING PYTHON DATASETS FOR DIRECT CODE TEXT PAYLOAD")
print("=" * 80)

for cand in CANDIDATES:
    print(f"\n[Testing: {cand['name']}]")
    print(f"Repo: {cand['repo']} | Subset: {cand['subset']}")
    try:
        kwargs = {"split": cand["split"], "streaming": True}
        if cand["subset"]:
            ds = load_dataset(cand["repo"], cand["subset"], **kwargs)
        else:
            ds = load_dataset(cand["repo"], **kwargs)

        sample = next(iter(ds))
        keys = list(sample.keys())
        print(f"  Available Columns: {keys}")

        found_col = None
        for k in cand["candidate_keys"]:
            if k in sample and isinstance(sample[k], str) and len(sample[k].strip()) > 0:
                found_col = k
                break

        if found_col:
            val = sample[found_col].strip()
            print(f"  Result: VALID (Found string data in '{found_col}' - {len(val):,} chars)")
            print("  --- Code Payload Preview (first 350 chars) ---")
            print("\n".join("    " + line for line in val[:350].splitlines()))
            print("  ----------------------------------------------")
        else:
            print("  Result: INVALID (No matched text column found)")
    except Exception as e:
        print(f"  Result: ERROR ({e})")

print("\n" + "=" * 80)
