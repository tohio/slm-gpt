import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

from transformer import ModelConfig, DecoderOnlyTransformer
from tokenizer import PretrainedTiktokenTokenizer


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_with_penalties(
    model: DecoderOnlyTransformer,
    idx: torch.Tensor,
    max_new_tokens: int,
    eot_token_id: int,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    stop_on_eot: bool = True,
) -> torch.Tensor:
    """
    Autoregressive generation with temperature, top-k, top-p, 
    repetition penalty, and clean EOT token stopping.
    """
    model.eval()

    for _ in range(max_new_tokens):
        # Crop context to max sequence length supported by positional embeddings
        idx_cond = idx[:, -model.config.max_seq_len:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # (B, V)

        # Apply repetition penalty to tokens already in context
        if repetition_penalty != 1.0:
            for token_id in set(idx[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # Scale by temperature
        logits = logits / max(temperature, 1e-5)

        # Top-K filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        # Top-P (Nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift right to keep first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Early termination if the model emits end-of-text
        if stop_on_eot and next_token.item() == eot_token_id:
            break

        idx = torch.cat((idx, next_token), dim=1)

    return idx


def main():
    parser = argparse.ArgumentParser(description="Sample from trained transformer checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/slm_model.pt")
    parser.add_argument("--prompt", type=str, default="The main reason why")
    parser.add_argument("--max_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--repetition_penalty", type=float, default=1.20)
    parser.add_argument("--stop_on_eot", action="store_true", default=True, help="Halt generation on <|endoftext|>")

    args = parser.parse_args()
    device = resolve_device()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading checkpoint '{checkpoint_path}' on {device}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    config = ModelConfig(**checkpoint["config"])
    model = DecoderOnlyTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded model architecture: {config.n_layers}L, {config.n_heads}H, {config.d_model}D")

    tokenizer = PretrainedTiktokenTokenizer("gpt2")
    eot_token_id = tokenizer.encode("<|endoftext|>")[0]

    prompt_tokens = tokenizer.encode(args.prompt)
    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    print(f"\n--- Generating (Prompt: '{args.prompt}', Temp: {args.temperature}, RepPenalty: {args.repetition_penalty}) ---")

    with torch.no_grad():
        out_tensor = sample_with_penalties(
            model=model,
            idx=x,
            max_new_tokens=args.max_tokens,
            eot_token_id=eot_token_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_on_eot=args.stop_on_eot,
        )

    out_tokens = out_tensor[0].tolist()

    # If <|endoftext|> was generated, truncate everything after it
    if eot_token_id in out_tokens[len(prompt_tokens):]:
        eot_idx = out_tokens.index(eot_token_id, len(prompt_tokens))
        out_tokens = out_tokens[:eot_idx]

    generated_text = tokenizer.decode(out_tokens)
    print(generated_text)
    print("\n--- End of Generation ---")


if __name__ == "__main__":
    main()