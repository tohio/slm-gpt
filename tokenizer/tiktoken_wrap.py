"""
tokenizer/tiktoken_wrap.py: Pretrained tiktoken wrapper with first-class special tokens
for ChatML, reasoning chains (<think>), tool calling, and FIM code infilling.
"""

from typing import Dict, List, Optional, Sequence, Union
import tiktoken
from .base import BaseTokenizer

# Canonical special tokens contract for slm-gpt (fits within 50304 vocab allocation)
DEFAULT_SPECIAL_TOKENS: Dict[str, int] = {
    "<|endoftext|>": 50256,
    "<|im_start|>": 50257,
    "<|im_end|>": 50258,
    "<|pad|>": 50259,
    "<s>": 50260,
    "</s>": 50261,
    "<|begin_of_text|>": 50262,
    "<think>": 50263,
    "</think>": 50264,
    "<tool_call>": 50265,
    "</tool_call>": 50266,
    "<tool_response>": 50267,
    "</tool_response>": 50268,
    "<|fim_prefix|>": 50269,
    "<|fim_middle|>": 50270,
    "<|fim_suffix|>": 50271,
    "<|fim_hole|>": 50272,
}

# Backward compatibility alias
DEFAULT_CHAT_SPECIAL_TOKENS = DEFAULT_SPECIAL_TOKENS


class PretrainedTiktokenTokenizer(BaseTokenizer):
    def __init__(
        self,
        encoding_name: str = "gpt2",
        extra_special_tokens: Optional[Dict[str, int]] = None,
    ):
        base_enc = tiktoken.get_encoding(encoding_name)

        merged_specials = dict(base_enc._special_tokens)
        merged_specials.update(DEFAULT_SPECIAL_TOKENS)
        if extra_special_tokens:
            merged_specials.update(extra_special_tokens)

        self.special_tokens = merged_specials
        self.enc = tiktoken.Encoding(
            name=f"{encoding_name}_slm",
            pat_str=base_enc._pat_str,
            mergeable_ranks=base_enc._mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        # Cache valid token IDs recognized by the underlying BPE engine
        # to safeguard against out-of-vocab/padding tokens (e.g. 50273..50303)
        self._valid_ids = set(base_enc._mergeable_ranks.values()) | set(self.special_tokens.values())

        # Cache standard special token IDs
        self._eot_id = self._safe_id("<|endoftext|>", 50256)
        self._im_start_id = self._safe_id("<|im_start|>", 50257)
        self._im_end_id = self._safe_id("<|im_end|>", 50258)
        self._pad_id = self._safe_id("<|pad|>", 50259)
        self._bos_id = self._safe_id("<s>", 50260)
        self._eos_id = self._safe_id("</s>", 50261)
        self._think_start_id = self._safe_id("<think>", 50263)
        self._think_end_id = self._safe_id("</think>", 50264)
        self._tool_call_start_id = self._safe_id("<tool_call>", 50265)
        self._tool_call_end_id = self._safe_id("</tool_call>", 50266)
        self._tool_resp_start_id = self._safe_id("<tool_response>", 50267)
        self._tool_resp_end_id = self._safe_id("</tool_response>", 50268)
        self._fim_prefix_id = self._safe_id("<|fim_prefix|>", 50269)
        self._fim_middle_id = self._safe_id("<|fim_middle|>", 50270)
        self._fim_suffix_id = self._safe_id("<|fim_suffix|>", 50271)
        self._fim_hole_id = self._safe_id("<|fim_hole|>", 50272)
        self._newline_id = self.enc.encode("\n")[0]

    def _safe_id(self, token_str: str, default: int) -> int:
        try:
            return self.enc.encode_single_token(token_str)
        except KeyError:
            return default

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
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def think_start_id(self) -> int:
        return self._think_start_id

    @property
    def think_end_id(self) -> int:
        return self._think_end_id

    @property
    def tool_call_start_id(self) -> int:
        return self._tool_call_start_id

    @property
    def tool_call_end_id(self) -> int:
        return self._tool_call_end_id

    @property
    def tool_resp_start_id(self) -> int:
        return self._tool_resp_start_id

    @property
    def tool_resp_end_id(self) -> int:
        return self._tool_resp_end_id

    @property
    def fim_prefix_id(self) -> int:
        return self._fim_prefix_id

    @property
    def fim_middle_id(self) -> int:
        return self._fim_middle_id

    @property
    def fim_suffix_id(self) -> int:
        return self._fim_suffix_id

    @property
    def fim_hole_id(self) -> int:
        return self._fim_hole_id

    @property
    def newline_id(self) -> int:
        return self._newline_id

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special="all")

    def decode(self, ids: Union[Sequence[int], int], errors: str = "replace") -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        elif isinstance(ids, (int, float)):
            ids = [int(ids)]

        # Filter out padding and unmapped vocabulary slots before delegating to tiktoken
        valid_ids = [int(i) for i in ids if int(i) in self._valid_ids]
        return self.enc.decode(valid_ids, errors=errors)