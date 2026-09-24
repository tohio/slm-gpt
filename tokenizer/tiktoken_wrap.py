"""tokenizer/tiktoken_wrap.py: Pretrained tiktoken wrapper with first-class ChatML special tokens."""

from typing import Dict, List, Optional
import tiktoken
from .base import BaseTokenizer

DEFAULT_CHAT_SPECIAL_TOKENS = {
    "<|endoftext|>": 50256,
    "<|im_start|>": 50257,
    "<|im_end|>": 50258,
}


class PretrainedTiktokenTokenizer(BaseTokenizer):
    def __init__(
        self,
        encoding_name: str = "gpt2",
        extra_special_tokens: Optional[Dict[str, int]] = None,
    ):
        base_enc = tiktoken.get_encoding(encoding_name)

        merged_specials = dict(base_enc._special_tokens)
        merged_specials.update(DEFAULT_CHAT_SPECIAL_TOKENS)
        if extra_special_tokens:
            merged_specials.update(extra_special_tokens)

        self.enc = tiktoken.Encoding(
            name=f"{encoding_name}_chatml",
            pat_str=base_enc._pat_str,
            mergeable_ranks=base_enc._mergeable_ranks,
            special_tokens=merged_specials,
        )

        self._eot_id = self.enc.encode_single_token("<|endoftext|>")
        self._im_start_id = self.enc.encode_single_token("<|im_start|>")
        self._im_end_id = self.enc.encode_single_token("<|im_end|>")
        self._newline_id = self.enc.encode("\n")[0]

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    @property
    def eot_id(self) -> int:
        return self._eot_id

    @property
    def im_start_id(self) -> int:
        return self._im_start_id

    @property
    def im_end_id(self) -> int:
        return self._im_end_id

    @property
    def newline_id(self) -> int:
        return self._newline_id

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special="all")

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)