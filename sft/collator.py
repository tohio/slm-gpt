"""
sft/collator.py: Hardware-aligned dynamic batch collator for SFT.
Pads sequences using dedicated <|pad|> (50259) and aligns sequence
lengths to multiples of 16 for NVIDIA Tensor Core efficiency.
"""

from typing import Dict, List
import torch
from sft.dataset import IGNORE_INDEX


class SFTDataCollator:
    """
    Collates variable-length dialogue tokens into dynamically padded,
    hardware-aligned batches.
    """

    def __init__(
        self,
        pad_token_id: int = 50259,  # Dedicated <|pad|> token
        pad_to_multiple_of: int = 16,  # Tensor Core alignment
    ):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(
        self, features: List[Dict[str, List[int]]]
    ) -> Dict[str, torch.Tensor]:
        # Determine maximum sequence length in this specific batch
        batch_max_len = max(len(x["input_ids"]) for x in features)

        # Round up to nearest multiple of 16 for Tensor Core alignment
        if self.pad_to_multiple_of > 0:
            batch_max_len = (
                (batch_max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for item in features:
            input_ids = item["input_ids"]
            labels = item["labels"]
            seq_len = len(input_ids)
            pad_len = batch_max_len - seq_len

            # Pad tokens with pad_token_id, labels with IGNORE_INDEX
            padded_inputs = input_ids + [self.pad_token_id] * pad_len
            padded_labels = labels + [IGNORE_INDEX] * pad_len
            attention_mask = [1] * seq_len + [0] * pad_len

            batch_input_ids.append(padded_inputs)
            batch_labels.append(padded_labels)
            batch_attention_mask.append(attention_mask)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(
                batch_attention_mask, dtype=torch.long
            ),
        }