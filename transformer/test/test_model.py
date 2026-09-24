import sys
from pathlib import Path
import torch

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from transformer.config import ModelConfig
from transformer.model import DecoderOnlyTransformer


def test_model_forward_and_loss():
    cfg = ModelConfig(
        vocab_size=256,
        max_seq_len=64,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0
    )
    model = DecoderOnlyTransformer(cfg)
    model.eval()

    B, T = 2, 8
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    # Test forward with loss (unpack logits, loss, kv_cache)
    logits, loss, _ = model(idx, targets=targets)
    assert logits.shape == (B, T, cfg.vocab_size), f"Expected {(B, T, cfg.vocab_size)}, got {logits.shape}"
    assert loss is not None and loss.item() > 0.0, "Loss calculation failed or returned non-positive value"
    print("✓ DecoderOnlyTransformer forward pass and loss verified.")


def test_weight_tying():
    cfg = ModelConfig(vocab_size=128, max_seq_len=32, d_model=32, n_heads=2, n_layers=1)
    model = DecoderOnlyTransformer(cfg)

    # Check that wte and lm_head point to the exact same tensor storage
    assert model.wte.weight.data_ptr() == model.lm_head.weight.data_ptr(), (
        "Weight tying failed: wte and lm_head do not share data pointers"
    )
    print("✓ Weight tying between embedding and lm_head verified.")


def test_generation():
    cfg = ModelConfig(vocab_size=128, max_seq_len=32, d_model=32, n_heads=2, n_layers=1)
    model = DecoderOnlyTransformer(cfg)
    model.eval()

    prompt = torch.tensor([[10, 20, 30]], dtype=torch.long)
    new_tokens = 5
    out = model.generate(prompt, max_new_tokens=new_tokens)

    assert out.shape == (1, 3 + new_tokens), f"Expected shape (1, 8), got {out.shape}"
    print("✓ Autoregressive generation loop verified.")


if __name__ == "__main__":
    test_model_forward_and_loss()
    test_weight_tying()
    test_generation()