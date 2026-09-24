"""tokenizer/base.py: Abstract base class defining the unified tokenizer contract."""

from abc import ABC, abstractmethod
from typing import List


class BaseTokenizer(ABC):
    """
    Contract for all tokenizers in slm-gpt.
    Guarantees that data collators, ChatML formatting, and causal generation
    operate identically regardless of the underlying tokenization backend.
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        pass

    @property
    @abstractmethod
    def eot_id(self) -> int:
        """Token ID for <|endoftext|>."""
        pass

    @property
    @abstractmethod
    def im_start_id(self) -> int:
        """Token ID for ChatML turn start (<|im_start|>)."""
        pass

    @property
    @abstractmethod
    def im_end_id(self) -> int:
        """Token ID for ChatML turn end (<|im_end|>)."""
        pass

    @property
    @abstractmethod
    def newline_id(self) -> int:
        """Token ID for newline character ('\\n')."""
        pass

    @property
    def pad_token_id(self) -> int:
        """Token ID used for dynamic batch padding. Defaults to eot_id."""
        return self.eot_id

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encodes string to a list of token integer IDs."""
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Decodes a list of token integer IDs back to a string."""
        pass