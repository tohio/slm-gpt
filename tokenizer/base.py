"""
tokenizer/base.py: Abstract base class defining the unified tokenizer contract.
Guarantees consistent token IDs, ChatML formatting, reasoning tags (<think>),
and dynamic batch padding across all tokenizer implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


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
        """Token ID for <|endoftext|> (50256)."""
        pass

    @property
    @abstractmethod
    def im_start_id(self) -> int:
        """Token ID for ChatML turn start (<|im_start|>, 50257)."""
        pass

    @property
    @abstractmethod
    def im_end_id(self) -> int:
        """Token ID for ChatML turn end (<|im_end|>, 50258)."""
        pass

    @property
    @abstractmethod
    def newline_id(self) -> int:
        """Token ID for newline character ('\\n')."""
        pass

    @property
    def pad_id(self) -> int:
        """Token ID used for dynamic batch padding (<|pad|>, 50259)."""
        return getattr(self, "_pad_id", self.eot_id)

    @property
    def pad_token_id(self) -> int:
        """Alias for pad_id ensuring compatibility across all collators."""
        return self.pad_id

    @property
    def bos_id(self) -> Optional[int]:
        """Token ID for beginning of stream (<s>, 50260)."""
        return getattr(self, "_bos_id", None)

    @property
    def eos_id(self) -> Optional[int]:
        """Token ID for end of stream (</s>, 50261)."""
        return getattr(self, "_eos_id", None)

    @property
    def think_start_id(self) -> Optional[int]:
        """Token ID for reasoning chain opening (<think>, 50263)."""
        return getattr(self, "_think_start_id", None)

    @property
    def think_end_id(self) -> Optional[int]:
        """Token ID for reasoning chain closure (</think>, 50264)."""
        return getattr(self, "_think_end_id", None)

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encodes string to a list of token integer IDs."""
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Decodes a list of token integer IDs back to a string."""
        pass