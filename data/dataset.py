from pathlib import Path
from typing import Union
import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, binary_path: Union[str, Path], seq_len: int, dtype: np.dtype = np.uint16):
        """
        Args:
            binary_path: Path to flat binary file of serialized token IDs.
            seq_len: Context window length T.
            dtype: Data type used when serializing tokens (typically uint16 for GPT-2 vocab).
        """
        self.binary_path = Path(binary_path)
        self.seq_len = seq_len
        self.dtype = dtype

        if not self.binary_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.binary_path}")

        # Open file as read-only memory map
        self.data = np.memmap(self.binary_path, dtype=self.dtype, mode="r")
        self.total_tokens = len(self.data)

        # We need (seq_len + 1) tokens to form a valid (x, y) pair
        assert self.total_tokens > self.seq_len, (
            f"Dataset has {self.total_tokens} tokens, but requires at least {self.seq_len + 1}"
        )

        # Number of non-overlapping or sliding sequences possible
        self.num_samples = self.total_tokens - self.seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        """
        Returns:
            x: torch.LongTensor of shape (seq_len,)
            y: torch.LongTensor of shape (seq_len,) shifted by +1 token
        """
        # Slice (seq_len + 1) tokens from disk
        chunk = self.data[idx : idx + self.seq_len + 1].astype(np.int64)

        # Convert to PyTorch tensors
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])

        return x, y
