import sys
from pathlib import Path
import torch

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from transformer.config import ModelConfig
from transformer.embedding import TransformerEmbedding


def test_embedding_shape_and_bounds():
    cfg = ModelConfig(vocab_size=1000, max_seq_len=64, d_model=32, n_heads=4, dropout=0.0)
    emb = TransformerEmbedding(cfg)
    emb.eval()

    B, T = 4, 16
    # Discrete token IDs within vocabulary range
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    out = emb(idx)

    assert out.shape == (B, T, cfg.d_model), f"Expected {(B, T, cfg.d_model)}, got {out.shape}"
    print("✓ TransformerEmbedding output shape verified.")


def test_embedding_max_seq_len_boundary():
    """Verify context length protection triggers when T exceeds max_seq_len."""
    cfg = ModelConfig(vocab_size=100, max_seq_len=8, d_model=16, n_heads=2)
    emb = TransformerEmbedding(cfg)

    # Sequence length 9 > max_seq_len 8
    overflow_idx = torch.randint(0, cfg.vocab_size, (1, 9))

    try:
        _ = emb(overflow_idx)
        assert False, "Failed to reject sequence length > max_seq_len"
    except AssertionError:
        pass

    print("✓ Context window boundary assertion verified.")


if __name__ == "__main__":
    test_embedding_shape_and_bounds()
    test_embedding_max_seq_len_boundary()