"""
tokenizer/bpe.py: Custom Byte-Pair Encoding (BPE) tokenizer with native special tokens
for ChatML, reasoning chains (<think>), tool calling, and FIM code infilling.
"""

import json
from typing import Dict, List, Optional, Set, Tuple
import regex as re

from .base import BaseTokenizer

GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

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


def bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class CustomBPETokenizer(BaseTokenizer):
    def __init__(self, special_tokens: Optional[Dict[str, int]] = None):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.pat = re.compile(GPT2_SPLIT_PATTERN)

        self.special_tokens: Dict[str, int] = (
            dict(special_tokens) if special_tokens is not None else dict(DEFAULT_SPECIAL_TOKENS)
        )
        self.inverse_special_tokens: Dict[int, str] = {
            v: k for k, v in self.special_tokens.items()
        }

        self.merges: Dict[Tuple[str, str], int] = {}
        self.encoder: Dict[str, int] = {}
        self.decoder: Dict[int, str] = {}

        self._build_special_regex()
        self._init_base_vocab()
        self._refresh_special_ids()

    def _build_special_regex(self):
        if self.special_tokens:
            sorted_specials = sorted(
                self.special_tokens.keys(), key=len, reverse=True
            )
            escaped = [re.escape(s) for s in sorted_specials]
            self.special_pattern = re.compile(f"({'|'.join(escaped)})")
        else:
            self.special_pattern = None

    def _init_base_vocab(self):
        self.encoder = {v: i for i, v in enumerate(self.byte_encoder.values())}
        self.decoder = {i: v for v, i in self.encoder.items()}

        for s_token, s_id in self.special_tokens.items():
            self.encoder[s_token] = s_id
            self.decoder[s_id] = s_token

    def _refresh_special_ids(self):
        """Resolves and caches special token IDs to satisfy BaseTokenizer contract."""
        self._eot_id = self.special_tokens.get("<|endoftext|>", 50256)
        self._im_start_id = self.special_tokens.get("<|im_start|>", 50257)
        self._im_end_id = self.special_tokens.get("<|im_end|>", 50258)
        self._pad_id = self.special_tokens.get("<|pad|>", 50259)
        self._bos_id = self.special_tokens.get("<s>", 50260)
        self._eos_id = self.special_tokens.get("</s>", 50261)
        self._think_start_id = self.special_tokens.get("<think>", 50263)
        self._think_end_id = self.special_tokens.get("</think>", 50264)
        self._tool_call_start_id = self.special_tokens.get("<tool_call>", 50265)
        self._tool_call_end_id = self.special_tokens.get("</tool_call>", 50266)
        self._tool_resp_start_id = self.special_tokens.get("<tool_response>", 50267)
        self._tool_resp_end_id = self.special_tokens.get("</tool_response>", 50268)
        self._fim_prefix_id = self.special_tokens.get("<|fim_prefix|>", 50269)
        self._fim_middle_id = self.special_tokens.get("<|fim_middle|>", 50270)
        self._fim_suffix_id = self.special_tokens.get("<|fim_suffix|>", 50271)
        self._fim_hole_id = self.special_tokens.get("<|fim_hole|>", 50272)

        nl_enc = self._encode_chunk("\n")
        if nl_enc:
            self._newline_id = nl_enc[0]
        else:
            self._newline_id = self.encoder.get(self.byte_encoder[ord("\n")], 10)

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

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

    def _get_stats(self, words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        counts: Dict[Tuple[str, str], int] = {}
        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        assert (
            vocab_size >= 256 + len(self.special_tokens)
        ), f"Vocab size must fit 256 bytes + {len(self.special_tokens)} special tokens."

        num_merges = vocab_size - 256 - len(self.special_tokens)
        words_raw = re.findall(self.pat, text)
        words = [
            [self.byte_encoder[b] for b in w.encode("utf-8")] for w in words_raw
        ]

        next_idx = 256
        for i in range(num_merges):
            stats = self._get_stats(words)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)

            while next_idx in self.inverse_special_tokens:
                next_idx += 1

            self.merges[best_pair] = i
            new_token = "".join(best_pair)
            self.encoder[new_token] = next_idx
            self.decoder[next_idx] = new_token
            next_idx += 1

            new_words = []
            for word in words:
                new_word = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and (word[j], word[j + 1]) == best_pair:
                        new_word.append(new_token)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                new_words.append(new_word)
            words = new_words

            if verbose and (i + 1) % 500 == 0:
                print(f"Merge {i + 1}/{num_merges}: {best_pair} -> '{new_token}'")

        self._refresh_special_ids()

    def _bpe(self, token_chars: List[str]) -> List[str]:
        word = token_chars
        if len(word) <= 1:
            return word

        while True:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            pair_to_merge = None
            min_rank = float("inf")
            for p in pairs:
                rank = self.merges.get(p, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    pair_to_merge = p

            if pair_to_merge is None or min_rank == float("inf"):
                break

            first, second = pair_to_merge
            new_word = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == first
                    and word[i + 1] == second
                ):
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) <= 1:
                break

        return word

    def encode(self, text: str) -> List[int]:
        if not self.special_pattern:
            return self._encode_chunk(text)

        tokens: List[int] = []
        parts = self.special_pattern.split(text)
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                tokens.append(self.special_tokens[part])
            else:
                tokens.extend(self._encode_chunk(part))
        return tokens

    def _encode_chunk(self, text_chunk: str) -> List[int]:
        bpe_tokens: List[int] = []
        for match in re.findall(self.pat, text_chunk):
            token_chars = [self.byte_encoder[b] for b in match.encode("utf-8")]
            merged = self._bpe(token_chars)
            bpe_tokens.extend([self.encoder[t] for t in merged])
        return bpe_tokens

    def decode(self, ids: List[int]) -> str:
        out_str = []
        byte_buf = []

        for idx in ids:
            if idx in self.inverse_special_tokens:
                if byte_buf:
                    out_str.append(
                        bytearray(byte_buf).decode("utf-8", errors="replace")
                    )
                    byte_buf = []
                out_str.append(self.inverse_special_tokens[idx])
            else:
                token_str = self.decoder.get(idx, "")
                byte_buf.extend([self.byte_decoder[c] for c in token_str if c in self.byte_decoder])

        if byte_buf:
            out_str.append(
                bytearray(byte_buf).decode("utf-8", errors="replace")
            )

        return "".join(out_str)

    def save(self, filepath: str):
        data = {
            "special_tokens": self.special_tokens,
            "encoder": self.encoder,
            "merges": [f"{p[0]} {p[1]}" for p in self.merges.keys()],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.special_tokens = data.get("special_tokens", DEFAULT_SPECIAL_TOKENS)
        self.inverse_special_tokens = {
            v: k for k, v in self.special_tokens.items()
        }
        self.encoder = data["encoder"]
        self.decoder = {int(v): k for k, v in self.encoder.items()}
        self.merges = {}
        for rank, line in enumerate(data["merges"]):
            p1, p2 = line.split(" ", 1)
            self.merges[(p1, p2)] = rank
        self._build_special_regex()
        self._refresh_special_ids()

    @classmethod
    def from_file(cls, filepath: str) -> "CustomBPETokenizer":
        tok = cls()
        tok.load(filepath)
        return tok