"""scaling/test/test_scaling.py: Unit tests for dynamic scaling solver and hardware profiler."""

import pytest
from scaling.engine import (
    count_parameters,
    derive_learning_rate,
    derive_model_config,
    format_size,
    parse_size_str,
)
from scaling.hardware import profile_hardware
from scaling.config import RuntimeConfig


def test_parse_size_str():
    assert parse_size_str("125M") == 125_000_000
    assert parse_size_str("350M") == 350_000_000
    assert parse_size_str("1B") == 1_000_000_000
    assert parse_size_str("1.5B") == 1_500_000_000
    assert parse_size_str(50_000_000) == 50_000_000


@pytest.mark.parametrize("target,expected_min,expected_max", [
    ("125M", 115_000_000, 135_000_000),
    ("350M", 320_000_000, 380_000_000),
    ("1B",   930_000_000, 1_070_000_000),
])
def test_derive_model_config_invariants(target, expected_min, expected_max):
    vocab_size = 50257
    cfg, actual_params = derive_model_config(target, vocab_size=vocab_size, gqa_ratio=3)

    # 1. Parameter budget tolerance
    assert expected_min <= actual_params <= expected_max

    # 2. Structural invariants
    assert cfg.d_model % 64 == 0, "d_model must be a multiple of 64"
    assert cfg.n_heads == cfg.d_model // 64, "Head dimension must be exactly 64"
    assert cfg.n_heads % 3 == 0, "n_heads must be cleanly divisible by GQA ratio (3)"
    assert cfg.n_kv_heads == cfg.n_heads // 3, "n_kv_heads must match GQA ratio"
    assert cfg.d_ffn % 256 == 0, "d_ffn must be aligned to multiple of 256"
    assert cfg.vocab_size % 64 == 0, "Vocab size must be padded to multiple of 64"

    # 3. Exact count matches parameter formula
    exact_count = count_parameters(cfg)
    assert exact_count == actual_params


def test_derive_learning_rate():
    max_lr_small, min_lr_small = derive_learning_rate(d_model=512)
    max_lr_large, min_lr_large = derive_learning_rate(d_model=2048)

    # Larger models must receive smaller base learning rates
    assert max_lr_small > max_lr_large
    assert min_lr_small == pytest.approx(max_lr_small * 0.1)
    assert min_lr_large == pytest.approx(max_lr_large * 0.1)


def test_hardware_profiler():
    profile = profile_hardware(max_seq_len=2048, d_model=768)
    assert profile.precision_str in ("bfloat16", "float16", "float32")
    assert profile.recommended_micro_batch >= 1


def test_runtime_config_dynamic_path():
    cfg = RuntimeConfig.build(size="350M", stage="pretrain")
    assert "350M" in cfg.output_dir or "3" in cfg.output_dir
    assert cfg.output_dir.startswith("checkpoints/pretrain_")
    assert cfg.model_cfg is not None