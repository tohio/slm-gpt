import sys
from pathlib import Path
import torch

# Add root folder to path so transformer can be imported
sys.path.append(str(Path(__file__).resolve().parents[2]))

from transformer.config import ModelConfig
from transformer.attention import CausalSelfAttention


def test_attention_shape():
    cfg = ModelConfig(d_model=64, n_heads=4, max_seq_len=128, dropout=0.0)
    attn = CausalSelfAttention(cfg)
    attn.eval()

    B, T, C = 2, 16, cfg.d_model
    x = torch.randn(B, T, C)
    out, _ = attn(x)

    assert out.shape == (B, T, C), f"Expected shape {(B, T, C)}, got {out.shape}"
    print("✓ Attention output shape verified.")


def test_attention_causality():
    """
    Changing tokens in the future (position t > 0) MUST NOT alter
    the attention output for past tokens (position t = 0).
    """
    cfg = ModelConfig(d_model=64, n_heads=4, max_seq_len=128, dropout=0.0)
    attn = CausalSelfAttention(cfg)
    attn.eval()

    # Create base input sequence
    x1 = torch.randn(1, 8, cfg.d_model)

    # Create second sequence identical at position 0, but modified at positions 1..7
    x2 = x1.clone()
    x2[:, 1:, :] = torch.randn(1, 7, cfg.d_model)

    with torch.no_grad():
        out1, _ = attn(x1)
        out2, _ = attn(x2)

    # Token at index 0 must produce identical outputs
    diff = (out1[:, 0, :] - out2[:, 0, :]).abs().max().item()
    assert diff < 1e-6, f"Causality leak detected! Difference at t=0: {diff}"
    print("✓ Causal masking verified (no future information leak).")


if __name__ == "__main__":
    test_attention_shape()
    test_attention_causality()