import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import TokenDataset


def create_dataloader(
    dataset: TokenDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
) -> DataLoader:
    """Standard PyTorch DataLoader wrapping TokenDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )


class FastTokenBatcher:
    """
    Direct random-access memory-map batcher.
    Avoids PyTorch DataLoader thread overhead when sampling random slices.
    """
    def __init__(self, binary_path: str, seq_len: int, batch_size: int, dtype: np.dtype = np.uint16):
        self.data = np.memmap(binary_path, dtype=dtype, mode="r")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_idx = len(self.data) - seq_len - 1

    def next_batch(self, device: torch.device = torch.device("cpu")):
        # Generate random start positions across the entire binary token stream
        ix = np.random.randint(0, self.max_idx, size=self.batch_size)

        # Gather batch
        x_list = [self.data[i : i + self.seq_len].astype(np.int64) for i in ix]
        y_list = [self.data[i + 1 : i + 1 + self.seq_len].astype(np.int64) for i in ix]

        x = torch.from_numpy(np.stack(x_list)).to(device)
        y = torch.from_numpy(np.stack(y_list)).to(device)
        return x, y
