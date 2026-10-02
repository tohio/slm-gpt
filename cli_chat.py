from transformer.env import get_model_tag
"""
cli_chat.py: Interactive command-line chat interface for slm-gpt.
Supports multi-turn ChatML conversations, streaming token generation,
graceful session clearing, and strict stop-token enforcement.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
from dataclasses import fields
from pathlib import Path
import sys
from typing import Dict, List, Optional, Set
import warnings

# Suppress FlashAttention / Cutlass JIT compilation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="nvidia_cutlass_dsl")
warnings.filterwarnings("ignore", message=".*Argument aux_data.*cannot be converted to a JitArgument.*")

import torch
import torch.nn.functional as F
import torch.serialization

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from tokenizer.factory import get_tokenizer
from transformer.config import ModelConfig
from transformer.model import DecoderOnlyTransformer

try:
    torch.serialization.add_safe_globals([ModelConfig])
except AttributeError:
    pass


def load_model(checkpoint_path: str, device: str) -> tuple[DecoderOnlyTransformer, ModelConfig]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"[Loading] Restoring weights from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    valid_fields = {f.name for f in fields(ModelConfig)}
    cfg_raw = checkpoint.get("config", {})
    if isinstance(cfg_raw, ModelConfig):
        cfg_dict = {f.name: getattr(cfg_raw, f.name) for f in fields(ModelConfig)}
    elif hasattr(cfg_raw, "__dict__"):
        cfg_dict = dict(cfg_raw.__dict__)
    else:
        cfg_dict = dict(cfg_raw)

    filtered = {k: v for k, v in cfg_dict.items() if k in valid_fields}
    config = ModelConfig(**filtered)

    model = DecoderOnlyTransformer(config).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    model.load_state_dict(state_dict, strict=False)

    # Tie embeddings
    model.lm_head.weight = model.wte.weight
    model.eval()

    return model, config


def build_chatml_prompt(
    history: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
) -> str:
    """Formats dialogue history into strict ChatML envelopes."""
    prompt = ""
    if system_prompt and system_prompt.strip():
        prompt += f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n"

    for turn in history:
        role = turn["role"]
        content = turn["content"].strip()
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    # Cue the assistant turn
    prompt += "<|im_start|>assistant\n"
    return prompt


@torch.no_grad()
def stream_generate(
    model: DecoderOnlyTransformer,
    tokenizer,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    stop_token_ids: Set[int],
    device: str,
    dtype: torch.dtype,
    max_seq_len: int,
) -> str:
    """
    Autoregressive KV-cached token generator with live terminal streaming,
    isolated repetition penalty (generated tokens only), and hard stop checks.
    """
    model.eval()
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # 1. Prefill prompt into KV caches
    use_autocast = (device == "cuda") and (dtype in (torch.float16, torch.bfloat16))
    with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
        logits, _, kv_caches = model(x)
        logits = logits[:, -1, :]  # Shape: (1, Vocab)

    generated_ids: List[int] = []
    generated_text = ""

    for _ in range(max_new_tokens):
        # 2. Isolated repetition penalty (only penalizes newly generated tokens, NEVER prompt or stop tokens)
        if repetition_penalty != 1.0 and len(generated_ids) > 0:
            for token_id in set(generated_ids):
                if token_id in stop_token_ids:
                    continue
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # 3. Temperature scaling
        logits = logits / max(temperature, 1e-5)

        # 4. Top-K filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        # 5. Top-P (Nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = -float("Inf")

        # 6. Sample next token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()

        # 7. Check stop tokens immediately before decoding or appending
        if token_id in stop_token_ids:
            break

        generated_ids.append(token_id)

        # 8. Live token decoding and streaming
        current_decoded = tokenizer.decode(generated_ids)
        new_token_str = current_decoded[len(generated_text):]
        generated_text = current_decoded
        sys.stdout.write(new_token_str)
        sys.stdout.flush()

        # Context boundary check
        if len(prompt_ids) + len(generated_ids) >= max_seq_len:
            break

        # 9. Autoregressive step with updated KV cache
        with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
            logits, _, kv_caches = model(next_token, kv_caches=kv_caches)
            logits = logits[:, -1, :]

    sys.stdout.write("\n\n")
    sys.stdout.flush()

    # Clean any residual stop sequences
    clean = generated_text.split("<|im_end|>")[0].replace("<|endoftext|>", "").strip()
    return clean


def run_chat(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    model, config = load_model(args.checkpoint, device)
    tokenizer = get_tokenizer(args.tokenizer_type, args.tokenizer_path)

    im_end_id = getattr(tokenizer, "im_end_id", 50258)
    eot_id = getattr(tokenizer, "eot_id", 50256)
    stop_tokens = {im_end_id, eot_id}

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized ({config.n_layers} layers, {total_params:,} parameters on {device.upper()} in {dtype}).")
    print("=" * 65)
    print("           slm-gpt Interactive Chat Interface              ")
    print(f" Checkpoint:  {args.checkpoint}")
    print(f" Device:      {device.upper()} ({dtype})")
    print(f" Sampling:    Temp={args.temperature} | Top-P={args.top_p} | Top-K={args.top_k} | Rep-Penalty={args.repetition_penalty}")
    print(f" Multi-Turn:  {'Disabled (--single_turn)' if args.single_turn else f'Active (Max {args.max_history_turns} turns)'}")
    print(" Commands:    'clear' to reset dialogue, 'exit' or 'quit' to end.")
    print("=" * 65 + "\n")

    history: List[Dict[str, str]] = []

    while True:
        try:
            user_input = input("User > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Session ended.")
            break

        if user_input.lower() == "clear":
            history.clear()
            print("Dialogue history cleared.\n")
            continue

        history.append({"role": "user", "content": user_input})

        # History pruning
        if args.single_turn:
            active_history = [history[-1]]
        else:
            # Retain only up to max_history_turns pairs
            active_history = history[-(args.max_history_turns * 2):]

        prompt_text = build_chatml_prompt(active_history, system_prompt=args.system_prompt)
        input_ids = tokenizer.encode(prompt_text)

        # Context safety limit
        max_prompt_len = config.max_seq_len - args.max_new_tokens
        if len(input_ids) > max_prompt_len:
            input_ids = input_ids[-max_prompt_len:]

        sys.stdout.write("Assistant > ")
        sys.stdout.flush()

        response = stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_token_ids=stop_tokens,
            device=device,
            dtype=dtype,
            max_seq_len=config.max_seq_len,
        )

        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive multi-turn CLI for slm-gpt.")
    parser.add_argument("--checkpoint", type=str, default=f"checkpoints/dpo_{get_model_tag()}/dpo_final.pt")
    parser.add_argument("--tokenizer_type", type=str, default="tiktoken")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--system_prompt", type=str, default="", help="Optional system prompt (default empty for small models).")
    parser.add_argument("--max_history_turns", type=int, default=3, help="Max conversational turns preserved in multi-turn mode.")
    parser.add_argument("--single_turn", action="store_true", help="Do not accumulate history across turns.")

    run_chat(parser.parse_args())