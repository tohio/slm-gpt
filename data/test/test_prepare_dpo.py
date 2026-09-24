"""
data/test/test_prepare_dpo.py: Tests DPO preference data normalization and filtering.
Validates the unified normalize_preference_record contract and local JSONL persistence.
"""

import json
from pathlib import Path
import sys
import tempfile
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from data.prepare_dpo import normalize_preference_record
from tokenizer.tiktoken_wrap import PretrainedTiktokenTokenizer


def test_normalize_conversational_record():
    """Verify multi-turn list format is normalized to standard prompt/chosen/rejected."""
    sample = {
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
    rec = normalize_preference_record(sample)
    assert rec is not None, "Failed to normalize conversational record"
    assert rec["system"] == "You are a concise assistant."
    assert rec["prompt"] == "What is the capital of France?"
    assert rec["chosen"] == "The capital of France is Paris."
    assert "London" in rec["rejected"]


def test_normalize_qa_record():
    """Verify flat prompt/chosen/rejected records are extracted."""
    sample = {
        "system": "Solve step-by-step.",
        "prompt": "What is 2 + 2?",
        "chosen": "2 + 2 = 4.",
        "rejected": "2 + 2 = 5.",
    }
    rec = normalize_preference_record(sample)
    assert rec is not None, "Failed to normalize QA record"
    assert rec["prompt"] == "What is 2 + 2?"
    assert rec["chosen"] == "2 + 2 = 4."
    assert rec["rejected"] == "2 + 2 = 5."


def test_filter_identical_completions():
    """Verify that identical chosen and rejected completions are discarded (returns None)."""
    identical_sample = {
        "prompt": "Hi",
        "chosen": "Hello!",
        "rejected": "Hello!",
    }
    assert normalize_preference_record(identical_sample) is None


def test_filter_length_exploits():
    """Verify that completions with extreme length disparity are rejected to prevent verbosity hacking."""
    exploit_sample = {
        "prompt": "Explain gravity.",
        "chosen": "Gravity is " + "a fundamental force that attracts objects " * 50,  # > 350 words, >> len(rejected)
        "rejected": "Gravity pulls things down.",
    }
    assert normalize_preference_record(exploit_sample) is None


def test_prepare_dpo_jsonl_roundtrip():
    """Verify normalized records write and parse correctly from JSONL."""
    tokenizer = PretrainedTiktokenTokenizer("gpt2")
    sample = {
        "prompt": "What is the speed of light?",
        "chosen": "Approximately 299,792,458 m/s.",
        "rejected": "About 100 miles per hour.",
    }
    norm = normalize_preference_record(sample)
    assert norm is not None

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "test_pairs.jsonl"
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(norm) + "\n")

        with open(tmp_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == 1
        assert len(tokenizer.encode(lines[0]["prompt"])) > 0
        assert len(tokenizer.encode(lines[0]["chosen"])) > 0