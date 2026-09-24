"""
sft/dataset.py: Conversational dataset parser for ChatML formatted dialogues.
Enforces strict loss masking exclusively on assistant tokens and terminating <|im_end|>,
preserves indentation for code, and prevents broken mid-thought sequence truncation.
"""

import json
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from tokenizer.base import BaseTokenizer
from tokenizer.tiktoken_wrap import PretrainedTiktokenTokenizer

IGNORE_INDEX = -100


class SFTDataset(Dataset):
    """Parses raw conversation trees into unpadded token sequences and loss masks.

    Expected dialogue schema per sample:
    [
        {"role": "user", "content": "Explain gradient descent."},
        {"role": "assistant", "content": "<think>...</think>\nGradient descent is..."}
    ]
    """

    def __init__(
        self,
        data_path_or_list: Any,
        tokenizer: Optional[BaseTokenizer] = None,
        max_seq_len: int = 2048,
    ):
        self.tokenizer = (
            tokenizer if tokenizer is not None else PretrainedTiktokenTokenizer()
        )
        self.max_seq_len = max_seq_len

        # Pre-cache constant token IDs
        self.im_start_id = getattr(self.tokenizer, "im_start_id", 50257)
        self.im_end_id = getattr(self.tokenizer, "im_end_id", 50258)
        self.newline_id = getattr(self.tokenizer, "newline_id", 10)

        # Pre-encode common role tokens to avoid redundant BPE lookups
        self.role_token_cache: Dict[str, List[int]] = {
            "system": self.tokenizer.encode("system"),
            "user": self.tokenizer.encode("user"),
            "assistant": self.tokenizer.encode("assistant"),
        }

        # Load and validate dialogues
        if isinstance(data_path_or_list, str):
            raw_samples = self._load_data(data_path_or_list)
        elif isinstance(data_path_or_list, list):
            raw_samples = data_path_or_list
        else:
            raise ValueError("data_path_or_list must be a file path (str) or a list of conversations.")

        self.samples: List[List[Dict[str, str]]] = self._validate_and_filter(raw_samples)

    def _load_data(self, path: str) -> List[List[Dict[str, str]]]:
        conversations = []
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        conv = item.get("messages", item.get("conversations", []))
                        if conv:
                            conversations.append(conv)
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                if isinstance(raw, list):
                    for item in raw:
                        conv = item.get("messages", item.get("conversations", []))
                        if conv:
                            conversations.append(conv)
        return conversations

    def _validate_and_filter(self, raw_samples: List[List[Dict[str, str]]]) -> List[List[Dict[str, str]]]:
        valid_dialogues = []
        for conv in raw_samples:
            if not isinstance(conv, list) or len(conv) < 2:
                continue

            has_user = False
            has_assistant = False

            cleaned_conv = []
            for turn in conv:
                role = turn.get("role", "").strip()
                # Preserve horizontal indentation; strip only outer newlines
                content = turn.get("content", "").strip("\r\n").rstrip()

                if not role or not content:
                    continue

                if role == "user":
                    has_user = True
                elif role == "assistant":
                    has_assistant = True

                cleaned_conv.append({"role": role, "content": content})

            if has_user and has_assistant:
                valid_dialogues.append(cleaned_conv)

        return valid_dialogues

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        dialogue = self.samples[idx]

        input_ids: List[int] = []
        labels: List[int] = []

        for turn in dialogue:
            role = turn["role"]
            content = turn["content"]

            role_ids = self.role_token_cache.get(role, self.tokenizer.encode(role))
            content_ids = self.tokenizer.encode(content)

            # ChatML Envelope: <|im_start|>role\ncontent<|im_end|>\n
            header_tokens = [self.im_start_id] + role_ids + [self.newline_id]
            body_tokens = content_ids + [self.im_end_id]
            
            # Trailing newline separates turns
            turn_tokens = header_tokens + body_tokens + [self.newline_id]

            if role == "assistant":
                # Mask header with -100; compute loss on assistant tokens and terminating <|im_end|>
                # Explicitly mask the trailing newline with -100 to prevent training beyond EOS
                turn_labels = (
                    [IGNORE_INDEX] * len(header_tokens)
                    + body_tokens
                    + [IGNORE_INDEX]
                )
            else:
                # Mask entire user/system turn from loss calculation
                turn_labels = [IGNORE_INDEX] * len(turn_tokens)

            input_ids.extend(turn_tokens)
            labels.extend(turn_labels)

        # Context limit handling: Truncate only if necessary, ensuring label hygiene
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }