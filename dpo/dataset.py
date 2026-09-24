"""
dpo/dataset.py: Pairwise preference dataset parser for ChatML format.
Masks prompt tokens and computes supervision loss exclusively on response tokens up to <|im_end|>.
"""

import json
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import Dataset

try:
    from tokenizer.tiktoken_wrap import PretrainedTiktokenTokenizer
except ImportError:
    from tokenizer.tiktoken_tokenizer import PretrainedTiktokenTokenizer

IGNORE_INDEX = -100


class PreferenceDataset(Dataset):
    """
    Expected input schema per sample:
    {
        "system": "Optional system prompt",
        "prompt": "User query here",
        "chosen": "High quality assistant response",
        "rejected": "Low quality or hallucinated response"
    }
    """

    def __init__(
        self,
        data_path_or_list: Any,
        tokenizer: Optional[PretrainedTiktokenTokenizer] = None,
        max_seq_len: int = 2048,
    ):
        self.tokenizer = tokenizer if tokenizer is not None else PretrainedTiktokenTokenizer()
        self.max_seq_len = max_seq_len

        if isinstance(data_path_or_list, str):
            self.samples = self._load_data(data_path_or_list)
        elif isinstance(data_path_or_list, list):
            self.samples = data_path_or_list
        else:
            raise ValueError("Data must be a filepath (str) or a list of dictionaries.")

        self.im_start_id = getattr(self.tokenizer, "im_start_id", 50257)
        self.im_end_id = getattr(self.tokenizer, "im_end_id", 50258)
        self.newline_id = getattr(self.tokenizer, "newline_id", 10)

    def _load_data(self, path: str) -> List[Dict[str, str]]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_chatml_pair(
        self, prompt: str, response: str, system: Optional[str] = None
    ) -> Tuple[List[int], List[int]]:
        input_ids: List[int] = []
        labels: List[int] = []

        # 1. System turn (if present) -> masked from loss
        if system and system.strip():
            sys_hdr = [self.im_start_id] + self.tokenizer.encode("system") + [self.newline_id]
            sys_body = self.tokenizer.encode(system.strip()) + [self.im_end_id] + [self.newline_id]
            sys_tokens = sys_hdr + sys_body
            input_ids.extend(sys_tokens)
            labels.extend([IGNORE_INDEX] * len(sys_tokens))

        # 2. User turn -> masked from loss
        usr_hdr = [self.im_start_id] + self.tokenizer.encode("user") + [self.newline_id]
        usr_body = self.tokenizer.encode(prompt.strip()) + [self.im_end_id] + [self.newline_id]
        usr_tokens = usr_hdr + usr_body
        input_ids.extend(usr_tokens)
        labels.extend([IGNORE_INDEX] * len(usr_tokens))

        # 3. Assistant turn -> header masked; response supervised; trailing \n masked
        asst_hdr = [self.im_start_id] + self.tokenizer.encode("assistant") + [self.newline_id]
        asst_body = self.tokenizer.encode(response.strip()) + [self.im_end_id]

        input_ids.extend(asst_hdr + asst_body + [self.newline_id])
        labels.extend([IGNORE_INDEX] * len(asst_hdr) + asst_body + [IGNORE_INDEX])

        # Truncate to maximum context window
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]

        return input_ids, labels

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        sample = self.samples[idx]
        system = sample.get("system", None)
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        chosen_ids, chosen_labels = self._encode_chatml_pair(prompt, chosen, system)
        rejected_ids, rejected_labels = self._encode_chatml_pair(prompt, rejected, system)

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_labels": rejected_labels,
        }