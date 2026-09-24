"""
cli_chat.py: Interactive command-line chat interface for slm-gpt.
Supports multi-turn ChatML conversations, streaming generation,
graceful session clearing, and strict stop-token enforcement.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
from dataclasses import fields
from pathlib import Path
import sys
from typing import Dict, List, Optional
import warnings

# Suppress FlashAttention / Cutlass JIT compilation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="nvidia_cutlass_dsl")
warnings.filterwarnings("ignore", message=".*Argument aux_data.*cannot be converted to a JitArgument.*")

import torch
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


def run_chat(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    model, config = load_model(args.checkpoint, device)
    tokenizer = get_tokenizer(args.tokenizer_type, args.tokenizer_path)

    im_end_id = getattr(tokenizer, "im_end_id", 50258)
    eot_id = getattr(tokenizer, "eot_id", 50256)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized ({config.n_layers} layers, {total_params:,} parameters on {device.upper()} in {dtype}).")
    print("=" * 65)
    print("           slm-gpt Interactive Chat Interface              ")
    print(f" Checkpoint: {args.checkpoint}")
    print(f" Device:     {device.upper()} ({dtype})")
    print(f" Sampling:   Temp={args.temperature} | Top-P={args.top_p} | Top-K={args.top_k} | Rep-Penalty={args.repetition_penalty}")
    print(" Commands:   'clear' to reset dialogue, 'exit' or 'quit' to end.")
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

        # Append current user query
        history.append({"role": "user", "content": user_input})

        # If single_turn is active, retain only the current user message
        active_history = [history[-1]] if args.single_turn else history
        prompt_text = build_chatml_prompt(active_history, system_prompt=args.system_prompt)
        input_ids = tokenizer.encode(prompt_text)

        # Context window safety truncate (retain trailing tokens if context overflows)
        if len(input_ids) >= config.max_seq_len - args.max_new_tokens:
            input_ids = input_ids[-(config.max_seq_len - args.max_new_tokens - 1):]

        x = torch.tensor([input_ids], dtype=torch.long, device=device)

        use_autocast = (device == "cuda") and (dtype in (torch.float16, torch.bfloat16))
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
                out = model.generate(
                    x,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    eot_token_id=im_end_id,
                )

        full_output = tokenizer.decode(out[0].tolist())

        # Extract only the newly generated text for the latest assistant turn
        assistant_turn = full_output.split("<|im_start|>assistant\n")[-1]
        clean_response = assistant_turn.split("<|im_end|>")[0].strip()

        # Remove any stray secondary EOS/EOT tokens
        clean_response = clean_response.replace("<|endoftext|>", "").strip()

        print(f"Assistant > {clean_response}\n")

        # Record cleanly into history with no token corruption
        history.append({"role": "assistant", "content": clean_response})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive multi-turn CLI for slm-gpt.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dpo_127M/dpo_final.pt")
    parser.add_argument("--tokenizer_type", type=str, default="tiktoken")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--system_prompt", type=str, default="You are a factual, concise assistant. Answer directly.")
    parser.add_argument("--single_turn", action="store_true", help="Do not accumulate history across turns.")

    run_chat(parser.parse_args())