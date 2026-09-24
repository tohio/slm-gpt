"""
dpo/collator.py: Dynamic dual-sequence batch collator for DPO.
Pads chosen and rejected token sequences to a uniform length aligned to multiples of 16.
"""

from typing import Dict, List
import torch

try:
    from .dataset import IGNORE_INDEX
except ImportError:
    from dpo.dataset import IGNORE_INDEX


class DPODataCollator:
    def __init__(self, pad_token_id: int = 50256, pad_to_multiple_of: int = 16):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # Determine maximum sequence length across both chosen and rejected completions
        max_chosen_len = max(len(x["chosen_input_ids"]) for x in features)
        max_rejected_len = max(len(x["rejected_input_ids"]) for x in features)
        batch_max_len = max(max_chosen_len, max_rejected_len)

        # Round up to nearest multiple of 16 for Tensor Core alignment
        if self.pad_to_multiple_of > 0:
            batch_max_len = (
                (batch_max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of

        chosen_inputs, chosen_labels = [], []
        rejected_inputs, rejected_labels = [], []

        for item in features:
            # Pad chosen sequences
            c_ids, c_lbls = item["chosen_input_ids"], item["chosen_labels"]
            c_pad = batch_max_len - len(c_ids)
            chosen_inputs.append(c_ids + [self.pad_token_id] * c_pad)
            chosen_labels.append(c_lbls + [IGNORE_INDEX] * c_pad)

            # Pad rejected sequences
            r_ids, r_lbls = item["rejected_input_ids"], item["rejected_labels"]
            r_pad = batch_max_len - len(r_ids)
            rejected_inputs.append(r_ids + [self.pad_token_id] * r_pad)
            rejected_labels.append(r_lbls + [IGNORE_INDEX] * r_pad)

        return {
            "chosen_input_ids": torch.as_tensor(chosen_inputs, dtype=torch.long),
            "chosen_labels": torch.as_tensor(chosen_labels, dtype=torch.long),
            "rejected_input_ids": torch.as_tensor(rejected_inputs, dtype=torch.long),
            "rejected_labels": torch.as_tensor(rejected_labels, dtype=torch.long),
        }