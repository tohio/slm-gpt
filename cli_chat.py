"""
cli_chat.py: Clean, standard interactive terminal chat interface for slm-gpt.

Adheres strictly to the slm-gpt model and tokenizer contracts:
- Full-context repetition penalty (prompt + generation) to prevent induction loops
- RoPE KV-cache stepping via forward(next_token, kv_caches=kv_caches)
- Native ChatML formatting matching sft/dataset.py
- PyTorch 2.6+ safe checkpoint deserialization
- Backward-compatible dynamic dtype derivation (bfloat16 for FlashAttention-4, float32 for CPU/MPS)
"""

import argparse
from contextlib import nullcontext
from dataclasses import fields
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import torch.serialization

try:
    from tokenizer.factory import get_tokenizer
except ImportError:
    try:
        from tokenizer.tiktoken_wrap import PretrainedTiktokenTokenizer as _Tok
        get_tokenizer = lambda t="tiktoken", p=None: _Tok()
    except ImportError:
        from tokenizer.tiktoken_tokenizer import PretrainedTiktokenTokenizer as _Tok
        get_tokenizer = lambda t="tiktoken", p=None: _Tok()

from transformer.config import ModelConfig
from transformer.model import DecoderOnlyTransformer

# Allowlist ModelConfig for PyTorch 2.6+ safe loading
try:
    torch.serialization.add_safe_globals([ModelConfig])
except AttributeError:
    pass

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="nvidia_cutlass_dsl")

def parse_args():
    parser = argparse.ArgumentParser(description="slm-gpt Interactive Chat CLI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/dpo_127M/dpo_final.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (0.0 = greedy)",
    )
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling cutoff")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus cutoff")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per turn",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.15,
        help="Repetition penalty applied over full sequence context",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="",
        help="Optional system prompt (default empty to match training distribution)",
    )
    parser.add_argument(
        "--single_turn",
        action="store_true",
        help="Reset context after each turn (evaluates queries in isolation)",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> Tuple[DecoderOnlyTransformer, ModelConfig]:
    """
    Restores model weights, enforces weight-tying invariants, and casts to target precision.
    Maintains the 2-tuple (model, cfg) signature for backward compatibility.
    """
    if not os.path.exists(checkpoint_path):
        # Fallback to SFT final checkpoint if DPO checkpoint is absent
        fallback = os.path.join(os.path.dirname(checkpoint_path).replace("dpo_", "sft_"), "sft_final.pt")
        if os.path.exists(fallback):
            print(f"[Notice] '{checkpoint_path}' not found. Falling back to '{fallback}'.")
            checkpoint_path = fallback
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"[Loading] Restoring weights from '{checkpoint_path}'...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    valid_field_names = {f.name for f in fields(ModelConfig)}
    cfg_raw = ckpt.get("config", {})

    if isinstance(cfg_raw, ModelConfig):
        cfg = cfg_raw
    elif isinstance(cfg_raw, dict) and cfg_raw:
        filtered = {k: v for k, v in cfg_raw.items() if k in valid_field_names}
        cfg = ModelConfig(**filtered)
    else:
        cfg = ModelConfig(
            vocab_size=50304,
            max_seq_len=2048,
            d_model=768,
            n_layers=14,
            n_heads=12,
            n_kv_heads=4,
            d_ffn=2048,
            dropout=0.0,
            bias=False,
        )

    cfg.dropout = 0.0

    # Determine execution precision
    is_cuda = "cuda" in str(device).lower() and torch.cuda.is_available()
    if is_cuda:
        target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        target_dtype = torch.float32

    model = DecoderOnlyTransformer(cfg)

    # Extract state dict across diverse checkpoint conventions
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)

    # Explicitly re-tie embeddings and cast to target precision
    model.lm_head.weight = model.wte.weight
    model = model.to(device=device, dtype=target_dtype)
    model.eval()

    assert model.wte.weight.data_ptr() == model.lm_head.weight.data_ptr(), (
        "Weight tying invariant broken: wte and lm_head do not share memory."
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized ({cfg.n_layers} layers, {total_params:,} parameters on {str(device).upper()} in {target_dtype}).")
    return model, cfg


def generate_stream(
    model: DecoderOnlyTransformer,
    tokenizer,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    device: str,
) -> List[int]:
    """
    Streams generated tokens to stdout using layerwise RoPE KV-caching.
    Applies full-context repetition penalty and dynamic autocast context.
    """
    model.eval()
    im_end_id = getattr(tokenizer, "im_end_id", 50257)
    eot_id = getattr(tokenizer, "eot_id", 50256)

    # Infer runtime dtype and backend dynamically from model weights
    dtype = next(model.parameters()).dtype
    device_str = str(device).lower()
    is_cuda = "cuda" in device_str and torch.cuda.is_available()
    use_autocast = is_cuda and (dtype in (torch.bfloat16, torch.float16))

    # Safe nullcontext fallback for CPU and Apple Silicon MPS
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if use_autocast else nullcontext()

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # 1. Prefill phase: compute initial KV caches for full prompt context
    with torch.no_grad(), autocast_ctx:
        logits, _, kv_caches = model(idx)
        logits = logits[:, -1, :].float()

    generated_ids: List[int] = []

    # 2. Sequential decode phase: 1 token per forward step
    for _ in range(max_new_tokens):
        # Full-context repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(idx[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # Greedy vs nucleus sampling
        if temperature <= 1e-4:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            scaled_logits = logits / max(temperature, 1e-5)

            if top_k > 0:
                v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                scaled_logits[scaled_logits < v[:, [-1]]] = -float("Inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                scaled_logits[:, indices_to_remove] = -float("Inf")

            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        token_val = next_token.item()

        # Stop condition: <|im_end|> or <|endoftext|>
        if token_val in (im_end_id, eot_id):
            break

        idx = torch.cat((idx, next_token), dim=1)
        generated_ids.append(token_val)

        # Stream decoded token chunk to terminal
        sys.stdout.write(tokenizer.decode([token_val]))
        sys.stdout.flush()

        if idx.size(1) >= model.config.max_seq_len:
            break

        # 3. Step forward single token with accumulated RoPE KV caches
        with torch.no_grad(), autocast_ctx:
            logits, _, kv_caches = model(next_token, kv_caches=kv_caches)
            logits = logits[:, -1, :].float()

    sys.stdout.write("\n")
    sys.stdout.flush()
    return generated_ids


def build_chatml_prompt(messages: List[Dict[str, str]], tokenizer) -> List[int]:
    """Encodes conversation turns into canonical ChatML tokens."""
    tokens = []
    for msg in messages:
        content = msg["content"].strip()
        if not content:
            continue
        role = msg["role"].strip()
        turn_str = f"<|im_start|>{role}\n{content}<|im_end|>\n"
        tokens.extend(tokenizer.encode(turn_str))

    asst_prefix = "<|im_start|>assistant\n"
    tokens.extend(tokenizer.encode(asst_prefix))
    return tokens


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = get_tokenizer("tiktoken")
    model, cfg = load_model(args.checkpoint, device)

    dtype = next(model.parameters()).dtype
    print("=" * 65)
    print("           slm-gpt Interactive Chat Interface              ")
    print(f" Checkpoint: {args.checkpoint}")
    print(f" Device:     {device.upper()} ({dtype})")
    print(f" Sampling:   Temp={args.temperature} | Top-P={args.top_p} | Top-K={args.top_k} | Rep-Penalty={args.repetition_penalty}")
    print(" Commands:   'clear' to reset dialogue, 'exit' or 'quit' to end.")
    print("=" * 65)

    messages: List[Dict[str, str]] = []
    if args.system_prompt.strip():
        messages.append({"role": "system", "content": args.system_prompt.strip()})

    while True:
        try:
            user_input = input("\nUser > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Session ended.")
            break
        if user_input.lower() == "clear":
            messages = []
            if args.system_prompt.strip():
                messages.append({"role": "system", "content": args.system_prompt.strip()})
            print("✓ Conversation history cleared.")
            continue

        if args.single_turn:
            messages = []
            if args.system_prompt.strip():
                messages.append({"role": "system", "content": args.system_prompt.strip()})

        messages.append({"role": "user", "content": user_input})
        prompt_ids = build_chatml_prompt(messages, tokenizer)

        # Context window truncation: slide window if approaching context limit
        while len(prompt_ids) >= cfg.max_seq_len - args.max_new_tokens and len(messages) > 1:
            drop_idx = 1 if messages[0]["role"] == "system" else 0
            messages.pop(drop_idx)
            prompt_ids = build_chatml_prompt(messages, tokenizer)

        sys.stdout.write("Assistant > ")
        sys.stdout.flush()

        response_token_ids = generate_stream(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
        )

        response_text = tokenizer.decode(response_token_ids)
        messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()