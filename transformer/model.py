import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .block import TransformerBlock


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding (RoPE replaces wpe absolute position table)
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (Press & Wolf, 2016)
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list | None]:
        # Token embedding only (no positional additions)
        x = self.drop(self.wte(idx))

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(x, kv_cache=layer_cache)
            if new_cache is not None:
                new_kv_caches.append(new_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, (new_kv_caches if new_kv_caches else None)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eot_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Fast O(1) per-token autoregressive generation with RoPE & layerwise Key-Value caching.
        """
        self.eval()

        # 1. Prefill phase: initialize KV caches with prompt
        logits, _, kv_caches = self(idx)
        logits = logits[:, -1, :]  # (B, V)

        # 2. Sequential generation: 1 token per forward step
        for _ in range(max_new_tokens):
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            logits = logits / max(temperature, 1e-5)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the sampled token to the generated sequence first
            idx = torch.cat((idx, next_token), dim=1)

            # Exit cleanly if stop token is reached
            if eot_token_id is not None and next_token.item() == eot_token_id:
                break

            if idx.size(1) >= self.config.max_seq_len:
                break

            # 3. Next step forward pass: 1 token with accumulated RoPE KV cache
            logits, _, kv_caches = self(next_token, kv_caches=kv_caches)
            logits = logits[:, -1, :]

        return idx