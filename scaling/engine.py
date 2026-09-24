"""
scaling/engine.py: Dynamic model dimension and hyperparameter derivation engine.
Derives transformer configurations and token budgets at runtime without static presets.
Enforces minimum depth invariants (min_layers=8) to prevent shallow sub-100M models.
"""

import math
import re
from typing import Tuple, Union

from transformer.config import ModelConfig


def parse_size_str(size_input: Union[str, int]) -> int:
    """Converts inputs like '70M', '125M', '350M', '1.2B', or integer parameter counts to an int."""
    if isinstance(size_input, int):
        return size_input

    s = str(size_input).strip().upper()
    match = re.match(r"^([\d.]+)\s*([KMB])?$", s)
    if not match:
        raise ValueError(
            f"Invalid size format: '{size_input}'. Expected format like '70M', '125M', '350M', '1B', or integer."
        )

    num_str, suffix = match.groups()
    val = float(num_str)
    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9, None: 1.0}
    return int(val * multipliers[suffix])


def format_size(param_count: int) -> str:
    """Formats exact integer parameter counts into readable tags (e.g., 71_456_832 -> '71M')."""
    if param_count >= 1e9:
        val = param_count / 1e9
        return f"{val:.2f}B".rstrip("0").rstrip(".") if val % 1 != 0 else f"{int(val)}B"
    elif param_count >= 1e6:
        return f"{round(param_count / 1e6)}M"
    elif param_count >= 1e3:
        return f"{round(param_count / 1e3)}K"
    return str(param_count)


def count_parameters(cfg: ModelConfig) -> int:
    """
    Computes exact parameter count of DecoderOnlyTransformer under tied embeddings.
    """
    # 1. Embedding / LM Head (tied)
    emb_params = cfg.vocab_size * cfg.d_model

    # 2. Final RMSNorm
    final_norm = cfg.d_model

    # 3. Per-layer parameters
    head_dim = cfg.d_model // cfg.n_heads
    q_proj = cfg.d_model * (cfg.n_heads * head_dim)
    k_proj = cfg.d_model * (cfg.n_kv_heads * head_dim)
    v_proj = cfg.d_model * (cfg.n_kv_heads * head_dim)
    o_proj = (cfg.n_heads * head_dim) * cfg.d_model
    attn_norms = 2 * cfg.d_model  # attn_norm + ffn_norm

    # SwiGLU MLP: w_gate, w_up, w_down
    mlp_params = 3 * (cfg.d_model * cfg.d_ffn)

    per_layer = q_proj + k_proj + v_proj + o_proj + attn_norms + mlp_params
    total_params = emb_params + final_norm + (cfg.n_layers * per_layer)
    return total_params


def derive_model_config(
    target_params: Union[str, int],
    vocab_size: int,
    max_seq_len: int = 2048,
    gqa_ratio: int = 3,
    min_layers: int = 8,
) -> Tuple[ModelConfig, int]:
    """
    Derives transformer dimensions dynamically to match a target parameter budget.
    Enforces min_layers=8 to prevent shallow degenerate architectures under heavy vocabularies.

    Invariants enforced:
      - Minimum depth >= min_layers (default: 8)
      - Head dimension = 64 (ideal for Tensor Core hardware alignment)
      - n_heads = d_model // 64
      - n_heads % gqa_ratio == 0 (clean integer KV-heads)
      - d_ffn = SwiGLU 8/3 * d_model rounded to nearest multiple of 256
      - d_model is a multiple of 64
      - vocab_size padded to nearest multiple of 64
    """
    budget = parse_size_str(target_params)
    padded_vocab = ((vocab_size + 63) // 64) * 64

    best_config = None
    min_diff = float("inf")
    best_actual_params = 0

    smallest_viable_cfg = None
    smallest_viable_params = float("inf")

    # Search space: standard transformer widths (multiples of 64)
    for d_model in range(192, 4096 + 1, 64):
        if d_model % 64 != 0:
            continue
        n_heads = d_model // 64

        if n_heads % gqa_ratio != 0:
            continue
        n_kv_heads = n_heads // gqa_ratio

        # SwiGLU hidden dim: ~ 8/3 * d_model aligned to 256
        d_ffn = int(round((8 / 3 * d_model) / 256)) * 256

        # Per-layer parameter calculation
        head_dim = 64
        q_params = d_model * (n_heads * head_dim)
        k_params = d_model * (n_kv_heads * head_dim)
        v_params = d_model * (n_kv_heads * head_dim)
        o_params = (n_heads * head_dim) * d_model
        layer_norms = 2 * d_model
        mlp_params = 3 * (d_model * d_ffn)
        per_layer = q_params + k_params + v_params + o_params + layer_norms + mlp_params

        emb_params = padded_vocab * d_model

        # Smallest viable candidate tracking with minimum layers
        min_layers_cfg = ModelConfig(
            vocab_size=padded_vocab,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_layers=min_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_ffn=d_ffn,
            dropout=0.0,
            bias=False,
        )
        min_p = count_parameters(min_layers_cfg)
        if min_p < smallest_viable_params:
            smallest_viable_params = min_p
            smallest_viable_cfg = min_layers_cfg

        rem_budget = budget - emb_params
        if rem_budget <= 0:
            continue

        # Solve for layer count respecting min_layers
        n_layers = max(min_layers, round(rem_budget / per_layer))

        # Check depth-to-width aspect ratio
        aspect = n_layers / d_model
        if not (0.010 <= aspect <= 0.040):
            continue

        candidate_cfg = ModelConfig(
            vocab_size=padded_vocab,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_ffn=d_ffn,
            dropout=0.0,
            bias=False,
        )

        actual_params = count_parameters(candidate_cfg)
        diff = abs(actual_params - budget)

        if diff < min_diff:
            min_diff = diff
            best_actual_params = actual_params
            best_config = candidate_cfg

    if best_config is None and smallest_viable_cfg is not None:
        print(
            f"[Scaling Notice] Target '{target_params}' is below the architectural floor for vocab {vocab_size}. "
            f"Rounding up to minimum viable architecture: {smallest_viable_params:,} params."
        )
        return smallest_viable_cfg, smallest_viable_params

    if best_config is None:
        raise RuntimeError(f"Could not derive valid architecture for target: {target_params}")

    return best_config, best_actual_params


def derive_learning_rate(d_model: int) -> Tuple[float, float]:
    """
    Derives base learning rate and cosine floor using width-based scaling.
    Base reference: d_model=768 -> max_lr=6e-4, min_lr=6e-5.
    Scales inversely with sqrt(d_model / 768).
    """
    scale = math.sqrt(768.0 / d_model)
    max_lr = 6e-4 * scale
    min_lr = max_lr * 0.1
    return max_lr, min_lr


def derive_training_budget(
    param_count: int,
    tokens_per_param: float = 50.0,
    max_seq_len: int = 2048,
    global_batch_size: int = 64,
) -> Tuple[int, int, int]:
    """
    Derives training token budget and optimizer step counts using scaling laws.
    Default ratio: 50:1 (50 tokens per parameter for SLM capacity maximization).
    Returns (target_tokens, total_steps, warmup_steps).
    """
    target_tokens = int(param_count * tokens_per_param)
    tokens_per_step = global_batch_size * max_seq_len
    total_steps = max(1, target_tokens // tokens_per_step)
    warmup_steps = min(2000, max(100, int(total_steps * 0.03)))
    return target_tokens, total_steps, warmup_steps