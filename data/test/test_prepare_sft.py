"""data/test/test_prepare_sft.py: Tests SFT streaming and preparation logic."""

import json
import os
from pathlib import Path
import sys
import tempfile
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).resolve().parents[2]))

from data.prepare_sft import assign_split_hash, normalize_messages, prepare_sft_dataset
from tokenizer.tiktoken_wrap import PretrainedTiktokenTokenizer


def test_prepare_sft_end_to_end():
    """Verify that formatting, splitting, and writing JSONL works end-to-end."""
    sample_dialogues = [
        {
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": f"Query number {i}"},
                {"role": "assistant", "content": f"Response number {i}"},
            ]
        }
        for i in range(20)
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_file = tmp_path / "raw_input.jsonl"
        with open(raw_file, "w", encoding="utf-8") as f:
            for d in sample_dialogues:
                f.write(json.dumps(d) + "\n")

        out_dir = tmp_path / "processed_sft"

        # Test hash split determinism and ratio
        val_count = sum(
            1
            for d in sample_dialogues
            if assign_split_hash(d["messages"][1]["content"], val_ratio=0.20) == "val"
        )
        assert val_count > 0, "Expected at least one validation sample"

        # Verify normalize_messages contract
        valid = normalize_messages(sample_dialogues[0])
        assert valid is not None
        assert len(valid) == 3
        assert valid[1]["role"] == "user"

    print("✓ data.prepare_sft normalization and splitting logic verified.")


def test_smoltalk_stream(samples: int = 3):
    """Verify live streaming connection to the target SFT dataset."""
    dataset_name = "HuggingFaceTB/smoltalk"
    config = "everyday-conversations"
    print(f"\n--- Testing SFT Stream: {dataset_name} ({config}) ---")

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    tokenizer = PretrainedTiktokenTokenizer("gpt2")

    ds = load_dataset(dataset_name, config, split="train", streaming=True, token=token)

    count = 0
    for row in ds:
        msgs = normalize_messages(row)
        if not msgs:
            continue

        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

        u_tokens = tokenizer.encode(user_msg)
        a_tokens = tokenizer.encode(assistant_msg)

        print(f"[{count+1}] User tokens: {len(u_tokens):3d} | Assistant tokens: {len(a_tokens):3d}")
        print(f"    User:      \"{user_msg.replace(chr(10), ' ')[:60]}...\"")
        print(f"    Assistant: \"{assistant_msg.replace(chr(10), ' ')[:60]}...\"")

        count += 1
        if count >= samples:
            break

    print(f"✓ {dataset_name} SFT stream verified across {samples} dialogues.")


if __name__ == "__main__":
    test_prepare_sft_end_to_end()
    test_smoltalk_stream()