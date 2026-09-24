from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    max_seq_len: int = 2048
    d_model: int = 256
    n_heads: int = 8
    n_kv_heads: int | None = None
    d_ffn: int | None = None
    n_layers: int = 6
    dropout: float = 0.1
    bias: bool = False

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads