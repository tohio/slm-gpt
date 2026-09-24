"""
data/test/test_prepare_sft.py: Unit tests for SFT data preparation pipeline.
Validates AST syntax checks, snippet extraction, and response sanitization.
"""

import json
from pathlib import Path
import pytest

from data.prepare_sft import (
    validate_python_ast,
    extract_python_snippet,
    sanitize_technical_response_preambles,
    sanitize_technical_response_postambles,
)

# Alias for backwards compatibility with any legacy test references
is_valid_python = validate_python_ast


class TestPythonASTValidation:
    """Tests strict forward-compatible Python AST validation."""

    def test_valid_python_syntax(self):
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        assert validate_python_ast(code) is True

    def test_invalid_python_syntax(self):
        code = "def broken_func(:\n    return 42"
        assert validate_python_ast(code) is False

    def test_empty_and_whitespace_code(self):
        assert validate_python_ast("") is False
        assert validate_python_ast("   \n\t  ") is False

    def test_strict_rejection_of_invalid_escape_sequences(self):
        """Ensures deprecated escape sequences (e.g. '\\c') fail validation under strict mode."""
        code_with_syntax_warning = 'pattern = "\\c"'
        assert validate_python_ast(code_with_syntax_warning) is False

    def test_valid_raw_string_escape_sequences(self):
        """Raw strings with regex escapes must pass AST validation."""
        valid_raw_code = 'import re\npattern = r"\c"'
        assert validate_python_ast(valid_raw_code) is True


class TestSnippetExtraction:
    """Tests extraction of Python code from markdown blocks and raw text."""

    def test_extract_from_fenced_markdown_python(self):
        text = "Here is the code:\n```python\ndef greet():\n    print('hello')\n```\nHope this helps!"
        expected = "def greet():\n    print('hello')"
        assert extract_python_snippet(text) == expected

    def test_extract_from_generic_fenced_markdown(self):
        text = "```\nx = [i for i in range(10)]\n```"
        expected = "x = [i for i in range(10)]"
        assert extract_python_snippet(text) == expected

    def test_extract_from_raw_valid_code(self):
        raw_code = "def square(x):\n    return x * x"
        assert extract_python_snippet(raw_code) == raw_code

    def test_extract_from_non_code_returns_none(self):
        prose = "This is simply an English paragraph describing how Python works without any valid code."
        assert extract_python_snippet(prose) is None


class TestResponseSanitization:
    """Tests iterative stripping of conversational preambles and postambles."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Sure! Here is the python code:\ndef run(): pass", "def run(): pass"),
            ("Certainly, here is the solution:\ndef solve(): pass", "def solve(): pass"),
            ("Here's the code:\ndef test(): pass", "def test(): pass"),
            ("Alright, below is the implementation:\nclass Node: pass", "class Node: pass"),
        ],
    )
    def test_sanitize_preambles(self, raw, expected):
        assert sanitize_technical_response_preambles(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("def run(): pass\nHope this helps!", "def run(): pass"),
            ("def run(): pass\nLet me know if you have any questions.", "def run(): pass"),
            ("def run(): pass\nHappy coding!", "def run(): pass"),
            ("def run(): pass\nGood luck!", "def run(): pass"),
        ],
    )
    def test_sanitize_postambles(self, raw, expected):
        assert sanitize_technical_response_postambles(raw) == expected