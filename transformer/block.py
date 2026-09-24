import torch
import torch.nn as nn

from .config import ModelConfig
from .attention import CausalSelfAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # Pre-LN Self-Attention with KV cache propagation
        normed_x = self.ln_1(x)
        attn_out, new_kv_cache = self.attn(normed_x, kv_cache=kv_cache)
        x = x + attn_out

        # Pre-LN Feed-Forward Network
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache