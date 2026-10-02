"""inspect_candidates.py: Inspect schema and actual payload of code datasets."""

from datasets import load_dataset


def probe_dataset(repo_id: str, split: str = "train", subset: str = None):
    print(f"\n{'='*80}")
    print(f"PROBING: {repo_id} (subset: {subset})")
    print(f"{'='*80}")
    try:
        kwargs = {"split": split, "streaming": True}
        if subset:
            ds = load_dataset(repo_id, subset, **kwargs)
        else:
            ds = load_dataset(repo_id, **kwargs)

        sample = next(iter(ds))
        print("Available Keys / Columns:", list(sample.keys()))

        # Find the text payload
        for key in ["text", "content", "code", "solution", "solutions", "prompt"]:
            if key in sample:
                val = str(sample[key])
                print(f"\n[Preview of key '{key}'] (first 500 chars):\n")
                print(val[:500])
                print("\n...")
                return

        # If none of the standard keys match, print the entire sample
        print("\nRaw sample structure:\n", sample)
    except Exception as e:
        print(f"Failed to probe {repo_id}: {e}")


# 1. BAAI/TACO
probe_dataset("BAAI/TACO")

# 2. ajibawa-2023/Python-Code-23k-ShareGPT
probe_dataset("ajibawa-2023/Python-Code-23k-ShareGPT")

# 3. BigCode The Stack v2 (smollm-corpus python-edu check)
probe_dataset("HuggingFaceTB/smollm-corpus", subset="python-edu")
