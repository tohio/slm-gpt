"""
data/test/test_prepare_sft.py: Unit tests for SFT dataset preparation and sanitation.
Validates preamble stripping, AST code validation, <think> formatting, and JSONL splitting.
"""

import json
from pathlib import Path
import sys
import tempfile
import pytest

# Ensure repository root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from data.prepare_sft_dataset import (
    extract_python_snippet,
    is_valid_python,
    sanitize_technical_response,
)
from tokenizer.tiktoken_wrap import PretrainedTiktokenTokenizer


def test_sanitize_technical_response_preambles():
    """Verify that sycophantic chat filler and introductory preambles are stripped."""
    dirty_samples = [
        ("Sure! I'd be happy to explain that.\nHere is the function:\ndef add(a, b): return a + b", "def add(a, b): return a + b"),
        ("Certainly! Here's the solution:\nimport math", "import math"),
        ("Great question! In this script, we solve it:\nx = 1", "x = 1"),
    ]
    for raw, expected in dirty_samples:
        cleaned = sanitize_technical_response(raw)
        assert cleaned == expected, f"Failed on '{raw}' -> got '{cleaned}'"


def test_sanitize_technical_response_postambles():
    """Verify that polite closing statements are stripped from technical solutions."""
    raw = "def solve():\n    return 42\n\nI hope this helps! Let me know if you have any questions."
    cleaned = sanitize_technical_response(raw)
    assert cleaned == "def solve():\n    return 42"


def test_python_ast_validation():
    """Verify that Python code snippets are checked for syntax correctness."""
    valid_code = "def is_even(n):\n    return n % 2 == 0"
    broken_code = "def is_even(n)\n    return n % 2 =="  # Missing colon & malformed syntax

    assert is_valid_python(valid_code) is True
    assert is_valid_python(broken_code) is False


def test_extract_python_snippet():
    """Verify markdown code block extraction."""
    markdown_text = (
        "Here is the code:\n"
        "```python\n"
        "def hello():\n"
        "    return 'world'\n"
        "```\n"
    )
    extracted = extract_python_snippet(markdown_text)
    assert extracted is not None
    assert "def hello():" in extracted
    assert is_valid_python(extracted) is True


def test_sft_jsonl_schema_and_tokenization():
    """Verify that SFT samples adhere to the ChatML schema and encode cleanly."""
    tokenizer = PretrainedTiktokenTokenizer("gpt2")

    sample_dialogue = {
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "<think>\n2 + 2 equals 4.\n</think>\n4"}
        ],
        "category": "math_cot"
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "test_sft.jsonl"
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample_dialogue) + "\n")

        with open(tmp_file, "r", encoding="utf-8") as f:
            loaded = [json.loads(line) for line in f]

        assert len(loaded) == 1
        msgs = loaded[0]["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert "<think>" in msgs[1]["content"]

        # Ensure tokenizer handles content without crashing
        u_toks = tokenizer.encode(msgs[0]["content"])
        a_toks = tokenizer.encode(msgs[1]["content"])
        assert len(u_toks) > 0
        assert len(a_toks) > 0