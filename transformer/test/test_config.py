import sys
from pathlib import Path

# Add 'slm-gpt' (the project root) to sys.path so 'transformer' can be imported
sys.path.append(str(Path(__file__).resolve().parents[2]))

from transformer.config import ModelConfig


def test_config():
    # 1. Default config check (d_model=256, n_heads=8 -> 32)
    cfg = ModelConfig()
    expected_default_head_dim = cfg.d_model // cfg.n_heads
    assert cfg.head_dim == expected_default_head_dim, (
        f"Expected default head_dim={expected_default_head_dim}, got {cfg.head_dim}"
    )

    # 2. Scaled 125M baseline check (d_model=768, n_heads=12 -> 64)
    scaled_cfg = ModelConfig(d_model=768, n_heads=12)
    assert scaled_cfg.head_dim == 64, f"Expected scaled head_dim=64, got {scaled_cfg.head_dim}"

    # 3. Invalid config assertion test (indivisible head dimension)
    try:
        _ = ModelConfig(d_model=768, n_heads=11)
        assert False, "Failed to catch non-divisible d_model / n_heads"
    except AssertionError:
        pass

    print("✓ ModelConfig passed assertions.")


if __name__ == "__main__":
    test_config()