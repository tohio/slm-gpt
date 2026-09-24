"""sft/dataset.py: Conversational dataset parser for ChatML formatted dialogues."""

import json
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from tokenizer.tiktoken_wrap import PretrainedTiktokenTokenizer

IGNORE_INDEX = -100


class SFTDataset(Dataset):
    """Parses raw conversation trees into unpadded token sequences and loss masks.

    Expected dialogue schema per sample:
    [
        {"role": "system", "content": "You are a concise AI assistant."},
        {"role": "user", "content": "Explain gradient descent."},
        {"role": "assistant", "content": "Gradient descent is an optimization algorithm..."}
    ]
    """

    def __init__(
        self,
        data_path_or_list: Any,
        tokenizer: Optional[PretrainedTiktokenTokenizer] = None,
        max_seq_len: int = 2048,
    ):
        self.tokenizer = (
            tokenizer if tokenizer is not None else PretrainedTiktokenTokenizer()
        )
        self.max_seq_len = max_seq_len

        # Load samples
        if isinstance(data_path_or_list, str):
            self.samples = self._load_data(data_path_or_list)
        elif isinstance(data_path_or_list, list):
            self.samples = data_path_or_list
        else:
            raise ValueError(
                "data_path_or_list must be a file path (str) or a list of conversations."
            )

        # Pre-cache constant token IDs
        self.im_start_id = self.tokenizer.im_start_id
        self.im_end_id = self.tokenizer.im_end_id
        self.newline_id = self.tokenizer.newline_id

    def _load_data(self, path: str) -> List[List[Dict[str, str]]]:
        conversations = []
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
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

    def __len__(self) -> int:
        return len(self.samples)

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        dialogue = self.samples[idx]

        input_ids: List[int] = []
        labels: List[int] = []

        for turn in dialogue:
            role = turn["role"].strip()
            content = turn["content"].strip()

            role_ids = self._encode(role)
            content_ids = self._encode(content)

            # ChatML format: <|im_start|>role\ncontent<|im_end|>\n
            header_tokens = [self.im_start_id] + role_ids + [self.newline_id]
            body_tokens = content_ids + [self.im_end_id] + [self.newline_id]
            turn_tokens = header_tokens + body_tokens

            if role in ("system", "user"):
                # Mask entire user/system turn from loss calculation
                turn_labels = [IGNORE_INDEX] * len(turn_tokens)
            elif role == "assistant":
                # Mask header; compute loss exclusively on assistant tokens + <|im_end|>\n
                header_mask = [IGNORE_INDEX] * len(header_tokens)
                turn_labels = header_mask + body_tokens
            else:
                turn_labels = [IGNORE_INDEX] * len(turn_tokens)

            input_ids.extend(turn_tokens)
            labels.extend(turn_labels)

        # Truncate sequence to max context window
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }