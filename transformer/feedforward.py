import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class FeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network (Shazeer, 2020 / LLaMA style).
    Replaces standard GELU MLP with bilinear gated SiLU activations:
        SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if hasattr(config, "d_ffn") and config.d_ffn is not None:
            hidden_dim = config.d_ffn
        else:
            hidden_dim = int(2 * (4 * config.d_model) / 3)
            hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.w_gate = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.w_up = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.w_down = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))