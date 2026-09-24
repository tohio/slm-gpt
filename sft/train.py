"""
sft/train.py: Supervised Fine-Tuning execution engine for slm-gpt.

Features:
- Loads architecture and weights directly from pre-trained checkpoint metadata.
- Native mixed precision (bfloat16 / float16 GradScaler auto-dispatch).
- Gradient accumulation with mathematically accurate step-loss telemetry.
- Prompt loss masking (ignore_index=-100) ensuring loss is strictly on assistant tokens.
- Validation perplexity evaluation loop.
- Qualitative Multi-Prompt Inference Checks evaluated every interval.
- Safe serialization using asdict(config) to guarantee PyTorch 2.6+ weights_only compatibility.
"""

from dataclasses import asdict, dataclass
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.serialization
from torch.utils.data import DataLoader, random_split

# slm-gpt internal modules
from transformer.config import ModelConfig
from transformer.model import DecoderOnlyTransformer as GPT
from sft.collator import SFTDataCollator
from sft.dataset import IGNORE_INDEX, SFTDataset
from tokenizer.factory import get_tokenizer

# Allowlist ModelConfig for PyTorch 2.6+
try:
    torch.serialization.add_safe_globals([ModelConfig])
except AttributeError:
    pass


@dataclass
class SFTArgs:
    # Model & Data Paths
    data_path: str = "data/sft/train.jsonl"
    val_path: Optional[str] = "data/sft/val.jsonl"
    pretrained_ckpt: str = "checkpoints/pretrained_125m/best_model.pt"
    output_dir: str = "checkpoints/sft_125m"

    # Tokenizer
    tokenizer_type: str = "tiktoken"
    tokenizer_path: Optional[str] = None

    # Optimization Hyperparameters
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8  # Effective batch size = 32 dialogues
    learning_rate: float = 2.5e-5
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    max_grad_norm: float = 1.0
    epochs: int = 3
    warmup_ratio: float = 0.05
    max_seq_len: int = 1024

    # Runtime, Checkpointing & Qualitative Inference
    eval_interval_steps: int = 50
    save_interval_epochs: int = 1
    log_interval_steps: int = 10
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


EVAL_PROMPTS = [
    "Hello! Who are you and what can you do?",
    "Explain the difference between a process and a thread in one concise sentence.",
    "Write a Python function to check if a word is a palindrome.",
    "If a train travels 60 miles per hour for 2.5 hours, how far does it go?",
]


def configure_precision(device: str) -> Tuple[torch.dtype, bool, Any]:
    """Detects optimal compute dtype and scaler requirements."""
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        use_scaler = False
        scaler = None
        print("[Precision] Using Native CUDA bfloat16 (GradScaler disabled).")
    elif device == "cuda":
        dtype = torch.float16
        use_scaler = True
        scaler = torch.amp.GradScaler("cuda")
        print("[Precision] Using CUDA float16 with GradScaler.")
    else:
        dtype = torch.float32
        use_scaler = False
        scaler = None
        print("[Precision] Using CPU float32.")
    return dtype, use_scaler, scaler


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Cosine learning rate scheduler with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def log_sample_generations(
    model: nn.Module,
    tokenizer,
    device: str,
    max_new_tokens: int = 64,
):
    """Executes multi-prompt qualitative inference checks with ChatML formatting."""
    model.eval()
    print("\n" + "=" * 70)
    print("                --- QUALITATIVE SFT INFERENCE CHECKS ---")
    print("=" * 70)

    im_end_str = "<|im_end|>"

    for i, user_prompt in enumerate(EVAL_PROMPTS, start=1):
        formatted_prompt = (
            f"<|im_start|>system\nYou are a helpful, concise AI assistant.<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        input_ids = tokenizer.encode(formatted_prompt)
        x = torch.tensor([input_ids], dtype=torch.long, device=device)

        out = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
        )

        full_output = tokenizer.decode(out[0].tolist())
        assistant_part = full_output.split("<|im_start|>assistant\n")[-1]

        stopped_cleanly = im_end_str in assistant_part
        clean_text = assistant_part.split(im_end_str)[0].strip()

        print(f"\n[{i}] Prompt: \"{user_prompt}\"")
        print(f"Assistant: {clean_text}")
        status = "✓ Stopped with <|im_end|>" if stopped_cleanly else "… (Context length cutoff)"
        print(f"Status:    {status}")

    print("=" * 70 + "\n")
    model.train()


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    compute_dtype: torch.dtype,
    max_eval_batches: int = 50,
) -> Tuple[float, float]:
    """Evaluates validation loss and active token perplexity."""
    model.eval()
    total_loss = 0.0
    total_active_tokens = 0

    use_autocast = compute_dtype in (torch.float16, torch.bfloat16) and (device == "cuda")

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_eval_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=compute_dtype, enabled=use_autocast):
                logits = model(input_ids)[0]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss_unreduced = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )

            active_tokens = (shift_labels != IGNORE_INDEX).sum().item()
            total_loss += loss_unreduced.item()
            total_active_tokens += active_tokens

    model.train()
    if total_active_tokens == 0:
        return 0.0, 0.0

    mean_loss = total_loss / total_active_tokens
    perplexity = math.exp(min(mean_loss, 20.0))
    return mean_loss, perplexity


def parse_args_from_cli(args_cls):
    """Loads default dataclass arguments and applies key=value CLI overrides."""
    kwargs = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            k = k.lstrip("-")
            if hasattr(args_cls, k):
                orig_val = getattr(args_cls, k)
                if isinstance(orig_val, bool):
                    kwargs[k] = v.lower() in ("true", "1", "yes")
                elif isinstance(orig_val, int):
                    kwargs[k] = int(v)
                elif isinstance(orig_val, float):
                    kwargs[k] = float(v)
                else:
                    kwargs[k] = v
    return args_cls(**kwargs)


def train(args: SFTArgs):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"--- Launching SFT Pipeline on {args.device.upper()} ---")

    # 1. Precision configuration
    compute_dtype, use_scaler, scaler = configure_precision(args.device)

    # 2. Tokenizer & Datasets
    tokenizer = get_tokenizer(args.tokenizer_type, args.tokenizer_path)
    collator = SFTDataCollator(pad_token_id=tokenizer.eot_id, pad_to_multiple_of=16)

    assert os.path.exists(args.data_path), f"SFT dataset not found at '{args.data_path}'."

    train_dataset = SFTDataset(args.data_path, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    if args.val_path and os.path.exists(args.val_path):
        val_dataset = SFTDataset(args.val_path, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    else:
        train_size = int(0.95 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=(args.device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=(args.device == "cuda"),
    )

    # 3. Model Initialization (Restore configuration and weights)
    assert os.path.exists(args.pretrained_ckpt), f"Pretrained checkpoint not found at: {args.pretrained_ckpt}"
    print(f"[Weights] Loading pre-trained base from '{args.pretrained_ckpt}'")
    checkpoint = torch.load(args.pretrained_ckpt, map_location=args.device, weights_only=False)

    if "config" in checkpoint:
        cfg_raw = checkpoint["config"]
        cfg_dict = cfg_raw.__dict__ if hasattr(cfg_raw, "__dict__") else dict(cfg_raw)
        cfg_dict["max_seq_len"] = args.max_seq_len
        config = ModelConfig(**cfg_dict)
    else:
        config = ModelConfig(
            vocab_size=50304,
            max_seq_len=args.max_seq_len,
            d_model=768,
            n_layers=14,
            n_heads=12,
            n_kv_heads=4,
            d_ffn=2048,
            dropout=0.0,
            bias=False,
            rope_theta=10000.0,
            tie_weights=True,
        )

    model = GPT(config).to(args.device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    assert model.wte.weight.data_ptr() == model.lm_head.weight.data_ptr(), "Weight tying broken!"
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Architecture] Initialized {config.n_layers} layers, {total_params:,} parameters ({total_params/1e6:.2f}M).")

    # 4. Optimizer & Schedule
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        fused=(args.device == "cuda"),
    )

    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_training_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_training_steps * args.warmup_ratio)

    scheduler = build_cosine_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_training_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    print(f"Total Epochs:          {args.epochs}")
    print(f"Per-Device Batch Size: {args.per_device_batch_size}")
    print(f"Gradient Accum Steps:  {args.gradient_accumulation_steps}")
    print(f"Total Optimizer Steps: {total_training_steps}")
    print(f"Warmup Steps:          {warmup_steps}")
    print("-" * 65)

    # 5. Training Loop
    global_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    running_loss = 0.0

    use_autocast = compute_dtype in (torch.float16, torch.bfloat16) and (args.device == "cuda")

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        for step_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(args.device, non_blocking=True)
            labels = batch["labels"].to(args.device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=compute_dtype, enabled=use_autocast):
                logits = model(input_ids)[0]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=IGNORE_INDEX,
                )
                loss_scaled = loss / args.gradient_accumulation_steps

            if use_scaler:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            accum_loss += loss_scaled.item()

            if (step_idx + 1) % args.gradient_accumulation_steps == 0 or (step_idx + 1) == len(train_loader):
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                running_loss += accum_loss
                accum_loss = 0.0

                if global_step % args.log_interval_steps == 0:
                    avg_step_loss = running_loss / args.log_interval_steps
                    curr_lr = scheduler.get_last_lr()[0]
                    print(
                        f"Epoch {epoch + 1:02d}/{args.epochs:02d} | "
                        f"Step {global_step:04d}/{total_training_steps:04d} | "
                        f"Loss: {avg_step_loss:.4f} | "
                        f"LR: {curr_lr:.2e}"
                    )
                    running_loss = 0.0

                if global_step % args.eval_interval_steps == 0:
                    val_loss, val_ppl = evaluate(model, val_loader, args.device, compute_dtype)
                    print(f"\n--> [Eval @ Step {global_step}] Val Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")
                    log_sample_generations(model, tokenizer, args.device)

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f}s")

        if (epoch + 1) % args.save_interval_epochs == 0 or (epoch + 1) == args.epochs:
            ckpt_name = f"sft_epoch_{epoch + 1}.pt"
            save_path = os.path.join(args.output_dir, ckpt_name)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": asdict(config),
                },
                save_path,
            )
            print(f"[Checkpoint] Saved model snapshot to: {save_path}")

    # Final policy checkpoint
    final_path = os.path.join(args.output_dir, "sft_final.pt")
    torch.save(
        {
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
        },
        final_path,
    )
    print(f"✓ SFT Run Completed. Final reference policy written to: {final_path}")


if __name__ == "__main__":
    train(parse_args_from_cli(SFTArgs()))