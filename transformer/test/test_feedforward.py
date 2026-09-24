import sys
from pathlib import Path
import torch

# Add root folder to path so transformer can be imported
sys.path.append(str(Path(__file__).resolve().parents[2]))

from transformer.config import ModelConfig
from transformer.feedforward import FeedForward


def test_feedforward_shape():
    # Explicitly pass n_heads=4 so 64 % 4 == 0 satisfies config validation
    cfg = ModelConfig(d_model=64, n_heads=4, dropout=0.0)
    ffn = FeedForward(cfg)
    ffn.eval()

    B, T, C = 2, 16, cfg.d_model
    x = torch.randn(B, T, C)
    out = ffn(x)

    assert out.shape == (B, T, C), f"Expected shape {(B, T, C)}, got {out.shape}"
    print("✓ FeedForward output shape verified.")


def test_feedforward_token_isolation():
    """
    Unlike attention, FeedForward operates on each token completely independently.
    Mutating token t=1 should have strictly 0 impact on token t=0.
    """
    cfg = ModelConfig(d_model=64, n_heads=4, dropout=0.0)
    ffn = FeedForward(cfg)
    ffn.eval()

    x1 = torch.randn(1, 4, cfg.d_model)
    x2 = x1.clone()
    x2[:, 1:, :] = torch.randn(1, 3, cfg.d_model)

    with torch.no_grad():
        out1 = ffn(x1)
        out2 = ffn(x2)

    diff = (out1[:, 0, :] - out2[:, 0, :]).abs().max().item()
    assert diff == 0.0, f"Token isolation violated! Discrepancy: {diff}"
    print("✓ FeedForward token-level independence verified.")


if __name__ == "__main__":
    test_feedforward_shape()
    test_feedforward_token_isolation()