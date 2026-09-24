"""scaling.py: Mathematical architecture & hyperparameter scaler for slm-gpt.

Generates optimal, hardware-aligned configurations for target parameter budgets
(e.g., 125M, 350M, 1B, 3B).
"""

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Dict, Optional


@dataclass
class ScaledConfig:
    # Model Architecture
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ffn: int
    d_head: int
    tie_weights: bool
    rope_theta: float

    # Exact Parameter Counts
    total_params: int
    embedding_params: int
    transformer_params: int

    # Recommended Pre-Training Hyperparameters
    peak_lr: float
    min_lr: float
    recommended_batch_tokens: int  # tokens per global gradient step
    chinchilla_target_tokens: int  # ~20x optimal token budget
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    grad_clip: float = 1.0


def round_to_multiple(val: int, multiple: int) -> int:
    return ((val + multiple - 1) // multiple) * multiple


def calculate_params(
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    d_ffn: int,
    tie_weights: bool,
) -> Dict[str, int]:
    d_head = d_model // n_heads

    # 1. Embeddings
    wte = vocab_size * d_model
    final_norm = d_model  # RMSNorm gain

    # 2. Per-Layer Attention: Q, K, V, Out + 2x RMSNorm
    w_q = d_model * (n_heads * d_head)  # d_model * d_model
    w_k = d_model * (n_kv_heads * d_head)
    w_v = d_model * (n_kv_heads * d_head)
    w_o = (n_heads * d_head) * d_model
    attn_norms = 2 * d_model  # Pre-attn norm + Pre-FFN norm
    per_block_attn = w_q + w_k + w_v + w_o + attn_norms

    # 3. Per-Layer SwiGLU: Gate, Up, Down
    w_gate = d_model * d_ffn
    w_up = d_model * d_ffn
    w_down = d_ffn * d_model
    per_block_ffn = w_gate + w_up + w_down

    per_layer = per_block_attn + per_block_ffn
    transformer_body = per_layer * n_layers

    # Head is tied to wte
    lm_head = 0 if tie_weights else (vocab_size * d_model)

    total = wte + transformer_body + final_norm + lm_head

    return {
        "total": total,
        "embedding": wte,
        "transformer_body": transformer_body,
    }


def derive_model_spec(
    target_params: int,
    max_seq_len: int = 2048,
    base_vocab_size: int = 50259,  # 50257 GPT-2 + <|im_start|>, <|im_end|>
    gqa_ratio: int = 3,  # Query-to-KV head ratio (e.g., 3:1 or 4:1)
    d_head: int = 64,  # Standard 64 (or 128 for >1B)
    tie_weights: bool = True,
    rope_theta: float = 10000.0,
) -> ScaledConfig:
    """Finds the optimal depth/width configuration closest to target_params

    satisfying all hardware alignment constraints.
    """
    # 1. Pad vocabulary to multiple of 128 for peak Tensor Core throughput
    vocab_size = round_to_multiple(base_vocab_size, 128)

    best_config: Optional[ScaledConfig] = None
    min_diff = float("inf")

    # Search space for typical balanced transformer aspect ratios
    # d_model must be divisible by (d_head * gqa_ratio)
    step_d_model = d_head * (
        gqa_ratio if gqa_ratio > 1 else 1
    )  # e.g., 64 * 3 = 192

    for d_model in range(384, 4096, step_d_model):
        n_heads = d_model // d_head
        if n_heads % gqa_ratio != 0:
            continue
        n_kv_heads = n_heads // gqa_ratio

        # SwiGLU 8/3 scaling, aligned to 256
        d_ffn = round_to_multiple(int(math.floor((8.0 / 3.0) * d_model)), 256)

        # Depth range centered around typical aspect ratio L / d_model ~ 0.010 to 0.025
        min_l = max(4, int(d_model * 0.010))
        max_l = max(min_l + 2, int(d_model * 0.025) + 6)

        for n_layers in range(min_l, max_l):
            counts = calculate_params(
                vocab_size,
                d_model,
                n_layers,
                n_heads,
                n_kv_heads,
                d_ffn,
                tie_weights,
            )
            diff = abs(counts["total"] - target_params)

            if diff < min_diff:
                min_diff = diff

                # Empirical learning rate scaling: baseline 6e-4 at 768 d_model (Gopher/LLaMA style)
                peak_lr = round(6e-4 * math.sqrt(768 / d_model), 6)
                min_lr = round(peak_lr * 0.1, 7)

                # Batch tokens scale mildly with parameters: ~0.5M tokens for small, up to 2M–4M at scale
                batch_tokens = (
                    round_to_multiple(
                        int(250_000 * math.sqrt(counts["total"] / 125_000_000)),
                        max_seq_len,
                    )
                    if counts["total"] >= 125_000_000
                    else 262_144
                )

                best_config = ScaledConfig(
                    vocab_size=vocab_size,
                    max_seq_len=max_seq_len,
                    d_model=d_model,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    d_ffn=d_ffn,
                    d_head=d_head,
                    tie_weights=tie_weights,
                    rope_theta=rope_theta,
                    total_params=counts["total"],
                    embedding_params=counts["embedding"],
                    transformer_params=counts["transformer_body"],
                    peak_lr=peak_lr,
                    min_lr=min_lr,
                    recommended_batch_tokens=batch_tokens,
                    chinchilla_target_tokens=counts["total"] * 20,
                )

    return best_config


if __name__ == "__main__":
    targets = [125_000_000, 350_000_000, 1_000_000_000]

    print(
        f"{'Target':<8} | {'Params':<12} | {'Layers':<6} | {'d_model':<7} | {'H_q':<4} | {'H_kv':<5} | {'d_ffn':<6} | {'Peak LR':<9} | {'Chinchilla Tokens'}"
    )
    print("-" * 95)

    for tgt in targets:
        # For ~1B, bump d_head to 128 (standard modern practice)
        d_head = 128 if tgt >= 1_000_000_000 else 64
        cfg = derive_model_spec(tgt, max_seq_len=2048, d_head=d_head)
        print(
            f"{tgt/1e6:<7.0f}M | {cfg.total_params/1e6:<10.2f}M | {cfg.n_layers:<6} | {cfg.d_model:<7} | {cfg.n_heads:<4} | {cfg.n_kv_heads:<5} | {cfg.d_ffn:<6} | {cfg.peak_lr:<9.2e} | {cfg.chinchilla_target_tokens/1e9:.2f}B"
        )