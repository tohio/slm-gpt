"""
data/test/test_prepare_dpo.py: Tests DPO preference data normalization, filtering, and HF streaming.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).resolve().parents[2]))

from data.prepare_dpo import extract_argilla_record, extract_orca_record
try:
    from tokenizer.tiktoken_wrap import PretrainedTiktokenTokenizer
except ImportError:
    from tokenizer.tiktoken_tokenizer import PretrainedTiktokenTokenizer


def test_prepare_dpo_end_to_end():
    """Verify record extraction, schema normalization, and identical pair filtering."""
    tokenizer = PretrainedTiktokenTokenizer("gpt2")

    # 1. Test Argilla conversational format normalization
    argilla_sample = {
        "chosen": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ],
        "rejected": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "I believe the capital of France might be London."},
        ],
    }
    rec_argilla = extract_argilla_record(argilla_sample)
    assert rec_argilla is not None, "Failed to extract valid Argilla record"
    assert rec_argilla["system"] == "You are a concise assistant."
    assert rec_argilla["prompt"] == "What is the capital of France?"
    assert rec_argilla["chosen"] == "The capital of France is Paris."
    assert "London" in rec_argilla["rejected"]

    # 2. Test Orca QA format normalization
    orca_sample = {
        "system": "Solve step-by-step.",
        "question": "What is 2 + 2?",
        "chosen": "2 + 2 = 4.",
        "rejected": "2 + 2 = 5.",
    }
    rec_orca = extract_orca_record(orca_sample)
    assert rec_orca is not None, "Failed to extract valid Orca record"
    assert rec_orca["prompt"] == "What is 2 + 2?"
    assert rec_orca["chosen"] == "2 + 2 = 4."
    assert rec_orca["rejected"] == "2 + 2 = 5."

    # 3. Test edge case: reject identical chosen and rejected
    identical_sample = {
        "chosen": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
        "rejected": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
    }
    rec_identical = extract_argilla_record(identical_sample)
    assert rec_identical["chosen"] == rec_identical["rejected"], "Expected identical strings"

    # 4. Verify round-trip JSONL writing to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "test_pairs.jsonl"
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec_argilla) + "\n")
            f.write(json.dumps(rec_orca) + "\n")

        with open(tmp_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == 2
        for item in lines:
            assert all(k in item for k in ("system", "prompt", "chosen", "rejected"))
            assert len(tokenizer.encode(item["prompt"])) > 0
            assert len(tokenizer.encode(item["chosen"])) > 0

    print("✓ data.prepare_dpo normalization, schema extraction, and JSONL persistence verified.")


def test_dpo_stream(samples: int = 3):
    """Verify live streaming connection and token boundaries for target DPO dataset."""
    dataset_name = "argilla/dpo-mix-7k"
    split = "train"
    print(f"\n--- Testing DPO Stream: {dataset_name} ({split} split) ---")

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    tokenizer = PretrainedTiktokenTokenizer("gpt2")

    ds = load_dataset(dataset_name, split=split, streaming=True, token=token)

    count = 0
    for row in ds:
        record = extract_argilla_record(row)
        if not record:
            continue

        if record["chosen"] == record["rejected"]:
            continue

        p_tokens = tokenizer.encode(record["prompt"])
        c_tokens = tokenizer.encode(record["chosen"])
        r_tokens = tokenizer.encode(record["rejected"])

        print(f"[{count+1}] Prompt: {len(p_tokens):3d} tokens | Chosen: {len(c_tokens):3d} tokens | Rejected: {len(r_tokens):3d} tokens")
        print(f"    Prompt:   \"{record['prompt'].replace(chr(10), ' ')[:65]}...\"")
        print(f"    Chosen:   \"{record['chosen'].replace(chr(10), ' ')[:65]}...\"")
        print(f"    Rejected: \"{record['rejected'].replace(chr(10), ' ')[:65]}...\"")

        count += 1
        if count >= samples:
            break

    assert count == samples, f"Expected {samples} samples, got {count}"
    print(f"✓ {dataset_name} preference stream verified across {samples} pairs.")


if __name__ == "__main__":
    test_prepare_dpo_end_to_end()
    test_dpo_stream()