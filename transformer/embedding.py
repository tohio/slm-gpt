import torch
import torch.nn as nn
from .config import ModelConfig


class TransformerEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Word Token Embeddings: maps vocabulary indices to d_model vectors
        self.wte = nn.Embedding(config.vocab_size, config.d_model)

        # Learned Positional Embeddings: maps position indices [0, max_seq_len-1] to d_model
        self.wpe = nn.Embedding(config.max_seq_len, config.d_model)

        # Regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: LongTensor of shape (batch_size, seq_len) with token indices in [0, vocab_size-1]
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        B, T = idx.size()

        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds maximum context window {self.config.max_seq_len}"
        )

        # Generate position indices [0, 1, 2, ..., T-1] on the same device as idx
        positions = torch.arange(0, T, dtype=torch.long, device=idx.device)  # Shape: (T,)

        # Compute embeddings
        tok_emb = self.wte(idx)        # Shape: (B, T, d_model)
        pos_emb = self.wpe(positions)  # Shape: (T, d_model), broadcasts over B

        # Combine token content with positional signal
        x = tok_emb + pos_emb
        return self.dropout(x)