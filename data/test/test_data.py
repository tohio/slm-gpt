import sys
from pathlib import Path
import tempfile
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from data.dataset import TokenDataset
from data.dataloader import create_dataloader, FastTokenBatcher


def test_token_dataset_alignment():
    seq_len = 8
    total_tokens = 100
    # Synthetic token sequence: 0, 1, 2, 3, ...
    tokens = np.arange(total_tokens, dtype=np.uint16)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tokens.tofile(tmp.name)

        dataset = TokenDataset(tmp.name, seq_len=seq_len, dtype=np.uint16)
        assert len(dataset) == total_tokens - seq_len

        # Test index 0
        x, y = dataset[0]
        assert x.shape == (seq_len,)
        assert y.shape == (seq_len,)

        # Strict target shift verification: y must equal x shifted by 1
        assert torch.equal(x, torch.arange(0, 8, dtype=torch.long))
        assert torch.equal(y, torch.arange(1, 9, dtype=torch.long))
        assert torch.equal(y[:-1], x[1:]), "Target y is not properly shifted relative to x"

        # Test DataLoader batching
        loader = create_dataloader(dataset, batch_size=4, shuffle=False)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (4, seq_len)
        assert batch_y.shape == (4, seq_len)

    print("✓ TokenDataset shift alignment and PyTorch DataLoader verified.")


def test_fast_batcher():
    seq_len = 16
    batch_size = 4
    tokens = np.random.randint(0, 1000, size=2000, dtype=np.uint16)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tokens.tofile(tmp.name)

        batcher = FastTokenBatcher(tmp.name, seq_len=seq_len, batch_size=batch_size, dtype=np.uint16)
        x, y = batcher.next_batch()

        assert x.shape == (batch_size, seq_len)
        assert y.shape == (batch_size, seq_len)
        # Check shift property on first sample of the batch
        assert torch.equal(y[0, :-1], x[0, 1:]), "FastBatcher target shift mismatch"

    print("✓ FastTokenBatcher random sampling and shape alignment verified.")


if __name__ == "__main__":
    test_token_dataset_alignment()
    test_fast_batcher()
