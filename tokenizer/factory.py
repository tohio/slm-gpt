"""tokenizer/factory.py: Central factory loader for dynamic tokenizer selection."""

from typing import Optional
from .base import BaseTokenizer
from .bpe import CustomBPETokenizer
from .tiktoken_wrap import PretrainedTiktokenTokenizer


def get_tokenizer(
    tokenizer_type: str = "tiktoken",
    tokenizer_path: Optional[str] = None,
    encoding_name: str = "gpt2",
) -> BaseTokenizer:
    """
    Central factory returning an instantiated tokenizer conforming to BaseTokenizer.

    Args:
        tokenizer_type: "tiktoken" or "custom"
        tokenizer_path: Path to custom BPE JSON artifact (required if tokenizer_type="custom")
        encoding_name: Base tiktoken encoding name (default: "gpt2")

    Returns:
        Instance conforming to BaseTokenizer.
    """
    choice = tokenizer_type.lower().strip()

    if choice in ("tiktoken", "gpt2"):
        return PretrainedTiktokenTokenizer(encoding_name=encoding_name)
    elif choice in ("custom", "bpe"):
        if not tokenizer_path:
            raise ValueError(
                "tokenizer_path (e.g. 'tokenizer/vocab.json') is required when using tokenizer_type='custom'."
            )
        return CustomBPETokenizer.from_file(tokenizer_path)
    else:
        raise ValueError(
            f"Unsupported tokenizer_type: '{tokenizer_type}'. Allowed choices: 'tiktoken', 'custom'."
        )