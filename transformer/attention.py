"""
transformer/attention.py: Grouped-Query Attention (GQA) with Dynamic Multi-Backend Dispatch.
Hierarchical Dispatch: FA4 (CuTeDSL sm90+/sm100) -> FA2 (Ampere/Ada sm80+) -> PyTorch Native SDPA.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

# =====================================================================
# Dynamic Hardware Kernel Resolution
# =====================================================================
HAS_FA4 = False
HAS_FA2 = False

FA4_FUNC = None
FA2_FUNC = None

if torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability()

    # Tier 1: Hopper (sm90) and Blackwell (sm100/sm120) -> FA4
    if major >= 9:
        try:
            from flash_attn.cute import flash_attn_func as fa4_f
            FA4_FUNC = fa4_f
            HAS_FA4 = True
        except (ImportError, Exception):
            try:
                from flash_attn_4 import flash_attn_func as fa4_f
                FA4_FUNC = fa4_f
                HAS_FA4 = True
            except (ImportError, Exception):
                pass

    # Tier 2: Ampere (sm80), Ada (sm89), or Hopper fallback -> FA2
    if not HAS_FA4 and major >= 8:
        try:
            from flash_attn import flash_attn_func as fa2_f
            FA2_FUNC = fa2_f
            HAS_FA2 = True
        except (ImportError, Exception):
            pass


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine frequency bands for RoPE across sequence positions."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
) -> torch.Tensor:
    """Apply rotary embeddings to tensor x of shape (B, num_heads, T, head_dim)."""
    B, H, T, D = x.size()
    cos_w = cos[offset : offset + T, :].to(x.device).to(x.dtype).unsqueeze(0).unsqueeze(0)
    sin_w = sin[offset : offset + T, :].to(x.device).to(x.dtype).unsqueeze(0).unsqueeze(0)

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    out1 = x1 * cos_w - x2 * sin_w
    out2 = x1 * sin_w + x2 * cos_w

    return torch.stack((out1, out2), dim=-1).flatten(-2)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads along the head dimension: (B, n_kv_heads, T, D) -> (B, n_heads, T, D)."""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


class CausalSelfAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) supporting:
    - Asymmetric Q vs KV heads (MHA / GQA / MQA)
    - Rotary Position Embeddings (RoPE)
    - O(1) Key-Value (KV) cache for autoregressive inference
    - Multi-tier kernel dispatch: FA4 -> FA2 -> PyTorch SDPA
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout_p = config.dropout

        assert self.head_dim % 2 == 0, "head_dim must be even for rotary position embeddings"

        # Precompute RoPE frequencies up to max_seq_len
        cos, sin = precompute_freqs_cis(self.head_dim, config.max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Projections
        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=config.bias)
        self.kv_proj = nn.Linear(config.d_model, 2 * self.n_kv_heads * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.resid_dropout = nn.Dropout(config.dropout)

        # Identify active backend
        if HAS_FA4:
            self.backend = "FlashAttention-4 (CuTeDSL sm90+/sm100)"
        elif HAS_FA2:
            self.backend = "FlashAttention-2 (Ampere/Ada sm80+)"
        else:
            self.backend = "PyTorch-Native-SDPA"

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size()

        # 1. Linear Projections
        q = self.q_proj(x)
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)

        # 2. Reshape to multi-head layout: (B, H, T, D)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE
        offset = kv_cache[0].size(2) if kv_cache is not None else 0
        q = apply_rotary_emb(q, self.cos, self.sin, offset=offset)
        k = apply_rotary_emb(k, self.cos, self.sin, offset=offset)

        # 4. KV Cache Update
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        new_kv_cache = (k, v) if not self.training else None

        # 5. Expand KV heads for GQA
        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        # 6. Attention Kernel Dispatch
        dropout_rate = self.dropout_p if self.training else 0.0

        if T == 1 and kv_cache is not None:
            # Single-token autoregressive decoding via SDPA
            y = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            y = y.transpose(1, 2).contiguous().view(B, T, C)

        elif HAS_FA4 and x.is_cuda and FA4_FUNC is not None:
            # FA4 layout: (batch, seqlen, nheads, headdim)
            q_fa = q.transpose(1, 2).contiguous()
            k_fa = k_rep.transpose(1, 2).contiguous()
            v_fa = v_rep.transpose(1, 2).contiguous()
            y = FA4_FUNC(q_fa, k_fa, v_fa, causal=True)
            if isinstance(y, tuple):
                y = y[0]
            y = y.contiguous().view(B, T, C)

        elif HAS_FA2 and x.is_cuda and FA2_FUNC is not None:
            # FA2 layout: (batch, seqlen, nheads, headdim)
            q_fa = q.transpose(1, 2).contiguous()
            k_fa = k_rep.transpose(1, 2).contiguous()
            v_fa = v_rep.transpose(1, 2).contiguous()
            y = FA2_FUNC(q_fa, k_fa, v_fa, dropout_p=dropout_rate, causal=True)
            if isinstance(y, tuple):
                y = y[0]
            y = y.contiguous().view(B, T, C)

        else:
            # Native PyTorch SDPA (CPU, MPS, or CUDA without FA packages)
            y = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                attn_mask=None,
                dropout_p=dropout_rate,
                is_causal=(T > 1),
            )
            y = y.transpose(1, 2).contiguous().view(B, T, C)

        out = self.resid_dropout(self.c_proj(y))
        return out, new_kv_cache