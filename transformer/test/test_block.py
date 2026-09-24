import sys
from pathlib import Path
import torch

# Add root folder to sys.path so transformer can be imported
sys.path.append(str(Path(__file__).resolve().parents[2]))

from transformer.config import ModelConfig
from transformer.block import TransformerBlock


def test_block_shape_and_residual_gradient():
    cfg = ModelConfig(d_model=64, n_heads=4, max_seq_len=128, dropout=0.0)
    block = TransformerBlock(cfg)

    B, T, C = 2, 16, cfg.d_model
    x = torch.randn(B, T, C, requires_grad=True)

    out, _ = block(x)
    assert out.shape == (B, T, C), f"Expected {(B, T, C)}, got {out.shape}"

    # Verify backprop through residual highway
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradient did not flow back to input x"
    assert not torch.isnan(x.grad).any(), "NaN gradients detected"
    print("✓ TransformerBlock shape and gradient backprop verified.")


def test_block_causality():
    """
    Ensure the full block (LayerNorm + Attention + MLP + Residuals)
    does not leak information from future tokens.
    """
    cfg = ModelConfig(d_model=64, n_heads=4, max_seq_len=128, dropout=0.0)
    block = TransformerBlock(cfg)
    block.eval()

    x1 = torch.randn(1, 8, cfg.d_model)
    x2 = x1.clone()
    x2[:, 1:, :] = torch.randn(1, 7, cfg.d_model)

    with torch.no_grad():
        out1, _ = block(x1)
        out2, _ = block(x2)

    diff = (out1[:, 0, :] - out2[:, 0, :]).abs().max().item()
    assert diff < 1e-6, f"Block causality violation! Discrepancy at t=0: {diff}"
    print("✓ TransformerBlock causal masking verified.")


if __name__ == "__main__":
    test_block_shape_and_residual_gradient()
    test_block_causality()