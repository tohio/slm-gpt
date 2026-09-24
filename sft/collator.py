"""
sft/collator.py: Hardware-aligned dynamic batch collator for SFT.
Pads sequences using dedicated <|pad|> (50259) and aligns sequence
lengths to multiples of 16 for NVIDIA Tensor Core efficiency.
"""

from typing import Dict, List, Optional
import torch
from sft.dataset import IGNORE_INDEX


class SFTDataCollator:
    """
    Collates variable-length dialogue tokens into dynamically padded,
    hardware-aligned batches using pre-allocated tensors.
    """

    def __init__(
        self,
        pad_token_id: int = 50259,  # Dedicated <|pad|> token
        pad_to_multiple_of: int = 16,  # Tensor Core alignment
        max_seq_len: Optional[int] = None,
    ):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_seq_len = max_seq_len

    def __call__(
        self, features: List[Dict[str, List[int]]]
    ) -> Dict[str, torch.Tensor]:
        if not features:
            return {}

        batch_size = len(features)
        batch_max_len = max(len(x["input_ids"]) for x in features)

        # Round up to nearest multiple of 16 for Tensor Core alignment
        if self.pad_to_multiple_of > 0:
            batch_max_len = (
                (batch_max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of

        # Guardrail against exceeding model positional embedding limits
        if self.max_seq_len is not None:
            batch_max_len = min(batch_max_len, self.max_seq_len)

        # Pre-allocate contiguous tensors directly
        batch_input_ids = torch.full(
            (batch_size, batch_max_len), self.pad_token_id, dtype=torch.long
        )
        batch_labels = torch.full(
            (batch_size, batch_max_len), IGNORE_INDEX, dtype=torch.long
        )
        batch_attention_mask = torch.zeros(
            (batch_size, batch_max_len), dtype=torch.long
        )

        for i, item in enumerate(features):
            input_ids = item["input_ids"][:batch_max_len]
            labels = item["labels"][:batch_max_len]
            seq_len = len(input_ids)

            batch_input_ids[i, :seq_len] = torch.tensor(input_ids, dtype=torch.long)
            batch_labels[i, :seq_len] = torch.tensor(labels, dtype=torch.long)
            batch_attention_mask[i, :seq_len] = 1

        return {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
            "attention_mask": batch_attention_mask,
        }