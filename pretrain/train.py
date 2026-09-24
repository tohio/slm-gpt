"""
pretrain/train.py: Production Distributed Pre-Training Engine for slm-gpt.
Reads binary memmapped tokens directly from sharded binaries.
Dynamically derives architecture and hyperparameters at runtime.
Supports multi-GPU via torchrun (DDP) with single-GPU fallback.
"""

from dataclasses import asdict
import glob
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from scaling.config import RuntimeConfig
from tokenizer.factory import get_tokenizer
from transformer import DecoderOnlyTransformer
from utils.distributed import (
    cleanup_distributed,
    init_distributed,
    is_main_process,
    print_rank0,
    reduce_tensor,
    save_checkpoint_rank0,
)
from utils.logger import WandbLogger

SAMPLE_PROMPTS = [
    "The solar system consists of",
    "In computer science, an algorithm is",
    "The history of science shows that",
]


# -----------------------------------------------------------------------------
# 1. Memory-Mapped Distributed DataLoader
# -----------------------------------------------------------------------------
class DistributedShardedDataLoader:
    def __init__(
        self,
        data_dir: str,
        split: str,
        batch_size: int,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.device = device

        single_bin = os.path.join(data_dir, f"{split}.bin")
        if os.path.exists(single_bin):
            self.shards = [single_bin]
        else:
            self.shards = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.bin")))

        assert len(self.shards) > 0, f"No binary files found for split '{split}' in {data_dir}"
        self.current_shard_idx = 0
        self.load_shard(self.current_shard_idx)

    def load_shard(self, shard_idx: int):
        self.current_shard_idx = shard_idx
        filename = self.shards[self.current_shard_idx]
        self.tokens = np.memmap(filename, dtype=np.uint16, mode="r")
        # Start each GPU at a disjoint batch offset
        self.current_pos = self.rank * (self.batch_size * self.seq_len)

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = self.batch_size, self.seq_len
        buf_len = B * T + 1
        step_stride = self.world_size * (B * T)

        if self.current_pos + buf_len > len(self.tokens):
            next_idx = (self.current_shard_idx + 1) % len(self.shards)
            self.load_shard(next_idx)

        buf = self.tokens[self.current_pos : self.current_pos + buf_len].astype(np.int64)
        x_np = buf[:-1].reshape(B, T)
        y_np = buf[1:].reshape(B, T)

        self.current_pos += step_stride

        x = torch.from_numpy(x_np).pin_memory().to(self.device, non_blocking=True)
        y = torch.from_numpy(y_np).pin_memory().to(self.device, non_blocking=True)
        return x, y


# -----------------------------------------------------------------------------
# 2. Learning Rate Scheduler & Optimizer Setup
# -----------------------------------------------------------------------------
def get_lr(it: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if it < warmup_steps:
        return max_lr * (it + 1) / max(1, warmup_steps)
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizers(model: nn.Module, weight_decay: float, learning_rate: float, betas=(0.9, 0.95)):
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=torch.cuda.is_available())


# -----------------------------------------------------------------------------
# 3. Live Sampling & Validation Loss
# -----------------------------------------------------------------------------
@torch.no_grad()
def log_sample_generation(raw_model: DecoderOnlyTransformer, enc, device: torch.device, max_tokens: int = 40):
    if not is_main_process():
        return

    raw_model.eval()
    print_rank0("\n" + "=" * 60)
    print_rank0("--- LIVE GENERATION SAMPLES ---")
    for prompt in SAMPLE_PROMPTS:
        x = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
        out = raw_model.generate(
            x,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        gen_text = enc.decode(out[0].tolist()).replace("\n", " ")
        print_rank0(f"\n[Prompt]: {prompt}")
        print_rank0(f"[Generated]: {gen_text}")
    print_rank0("=" * 60 + "\n")
    raw_model.train()


@torch.no_grad()
def estimate_loss(model: nn.Module, val_loader: DistributedShardedDataLoader, dtype: torch.dtype, eval_iters: int = 20) -> float:
    model.eval()
    losses = torch.zeros(eval_iters, device=val_loader.device)
    use_autocast = dtype in (torch.float16, torch.bfloat16)
    
    for k in range(eval_iters):
        x, y = val_loader.next_batch()
        with torch.autocast(device_type=val_loader.device.type, dtype=dtype, enabled=use_autocast):
            _, loss, _ = model(x, targets=y)
        losses[k] = loss.detach()

    local_mean = losses.mean()
    global_val_loss = reduce_tensor(local_mean, average=True).item()
    model.train()
    return float(global_val_loss)


def parse_cli_args(argv: List[str]) -> Dict[str, Any]:
    out = {}
    for arg in argv:
        if "=" in arg:
            k, v = arg.split("=", 1)
            k = k.lstrip("-")
            if v.isdigit():
                out[k] = int(v)
            else:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


# -----------------------------------------------------------------------------
# 4. Main Training Routine
# -----------------------------------------------------------------------------
def main():
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1337)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    rank, local_rank, world_size, is_distributed, device = init_distributed()

    cli_kwargs = parse_cli_args(sys.argv[1:])
    size_target = cli_kwargs.pop("size", "125M")
    data_dir = cli_kwargs.pop("data_dir", "data/pretrain")
    tok_type = cli_kwargs.pop("tokenizer_type", "tiktoken")
    tok_path = cli_kwargs.pop("tokenizer_path", None)
    eval_interval = cli_kwargs.pop("eval_interval", 1000)
    eval_iters = cli_kwargs.pop("eval_iters", 20)
    save_interval = cli_kwargs.pop("save_interval", 1000)
    log_interval = cli_kwargs.pop("log_interval", 10)
    warmup_steps = cli_kwargs.pop("warmup_steps", 500)
    grad_clip = cli_kwargs.pop("grad_clip", 1.0)
    user_max_steps = cli_kwargs.pop("max_steps", None)

    tokenizer = get_tokenizer(tok_type, tok_path)

    # Dynamically derive architecture, learning rates, batching, and directories
    cfg = RuntimeConfig.build(
        size=size_target,
        vocab_size=tokenizer.vocab_size,
        stage="pretrain",
        world_size=world_size,
        tokenizer_type=tok_type,
        tokenizer_path=tok_path,
        **cli_kwargs,
    )

    dtype = cfg.hardware.precision
    config = cfg.model_cfg

    print_rank0("=" * 65)
    print_rank0("         slm-gpt Dynamic Distributed Pre-training Engine       ")
    print_rank0("=" * 65)
    print_rank0(f"Target Size Tag:        {cfg.size} -> Resolved: {cfg.size_tag} ({cfg.actual_params:,} params)")
    print_rank0(f"Data Directory:         {data_dir}")
    print_rank0(f"Output Directory:       {cfg.output_dir}")
    print_rank0(f"Mode:                   {'Multi-GPU DDP' if is_distributed else 'Single-GPU / Local'}")
    print_rank0(f"Cluster World Size:     {world_size} (Local Rank: {local_rank})")
    print_rank0(f"Device & Precision:     {device} | {cfg.hardware.precision_str}")
    print_rank0(f"Sequence Length (T):    {config.max_seq_len}")
    print_rank0(f"Micro-batch / GPU:      {cfg.micro_batch_size} | Global Batch Target: {cfg.global_batch_size}")
    print_rank0(f"Grad Accumulation:      {cfg.grad_accum_steps} micro-steps")
    print_rank0(f"Learning Rate:          Peak: {cfg.max_lr:.2e} -> Min: {cfg.min_lr:.2e} (Warmup: {warmup_steps})")
    print_rank0(f"Architecture:           Layers={config.n_layers}, d_model={config.d_model}, GQA={config.n_heads}:{config.n_kv_heads}, d_ffn={config.d_ffn}")
    print_rank0("-" * 65)

    if is_main_process():
        os.makedirs(cfg.output_dir, exist_ok=True)
        cfg.save_json(os.path.join(cfg.output_dir, "resolved_config.json"))

    # Initialize Dataclass-driven W&B logger (reads .env credentials on rank 0)
    logger = WandbLogger(runtime_cfg=cfg, run_name=f"{cfg.size_tag}-pretrain")

    # Instantiate model and verify weight tying
    model = DecoderOnlyTransformer(config).to(device)
    raw_model = model
    assert raw_model.wte.weight.data_ptr() == raw_model.lm_head.weight.data_ptr(), (
        "Weight tying invariant broken: wte and lm_head do not share memory!"
    )
    unique_params = sum(p.numel() for p in raw_model.parameters())
    print_rank0(f"Verified Model Storage: {unique_params:,} parameters ({unique_params / 1e6:.2f}M)")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        raw_model = model.module

    # Initialize distributed memmapped dataloaders
    train_loader = DistributedShardedDataLoader(
        data_dir, "train", cfg.micro_batch_size, config.max_seq_len, rank=rank, world_size=world_size, device=device
    )
    val_loader = DistributedShardedDataLoader(
        data_dir, "val", cfg.micro_batch_size, config.max_seq_len, rank=rank, world_size=world_size, device=device
    )

    tokens_per_opt_step = cfg.micro_batch_size * world_size * cfg.grad_accum_steps * config.max_seq_len
    total_tokens = len(train_loader.tokens)
    max_steps = user_max_steps if user_max_steps is not None else (total_tokens // tokens_per_opt_step)

    print_rank0(f"Total Dataset Tokens:   {total_tokens:,} ({total_tokens / 1e9:.2f}B)")
    print_rank0(f"Batch Tokens / Step:    {tokens_per_opt_step:,} (Micro: {cfg.micro_batch_size}, Accum: {cfg.grad_accum_steps})")
    print_rank0(f"Total Optimizer Steps:  {max_steps:,} (Warmup: {warmup_steps})")

    optimizer = configure_optimizers(raw_model, cfg.weight_decay, cfg.max_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and device.type == "cuda"))

    best_val_loss = float("inf")
    tokens_processed = 0
    t0 = time.time()

    model.train()
    for step in range(max_steps + 1):
        # Periodic Evaluation & Live Text Samples
        if step % eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, dtype=dtype, eval_iters=eval_iters)
            ppl = math.exp(min(val_loss, 20.0))
            print_rank0(f"\n[Step {step}/{max_steps}] Validation Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")
            log_sample_generation(raw_model, tokenizer, device=device, max_tokens=40)

            logger.log({"eval/loss": val_loss, "eval/perplexity": ppl}, step=step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = os.path.join(cfg.output_dir, "best_model.pt")
                save_checkpoint_rank0(
                    {
                        "step": step,
                        "model_state_dict": raw_model.state_dict(),
                        "config": asdict(config),
                        "val_loss": val_loss,
                        "runtime_meta": {
                            "size_tag": cfg.size_tag,
                            "tokenizer_type": tok_type,
                            "tokenizer_path": tok_path,
                            "vocab_size": tokenizer.vocab_size,
                        },
                    },
                    best_ckpt_path,
                )
                print_rank0(f"Saved new best checkpoint to {best_ckpt_path}")

        if step == max_steps:
            break

        # Learning Rate Step
        lr = get_lr(step, warmup_steps, max_steps, cfg.max_lr, cfg.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient Accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(cfg.grad_accum_steps):
            x, y = train_loader.next_batch()

            is_sync_step = (micro_step == cfg.grad_accum_steps - 1)
            sync_context = model.no_sync if (is_distributed and not is_sync_step) else lambda: torch.enable_grad()
            use_autocast = dtype in (torch.float16, torch.bfloat16)

            with sync_context():
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
                    _, loss, _ = model(x, targets=y)
                    loss = loss / cfg.grad_accum_steps

                accum_loss += loss.detach()

                if dtype == torch.float16 and device.type == "cuda":
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

        # Gradient Clipping & Optimizer Step
        if dtype == torch.float16 and device.type == "cuda":
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            norm = nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
            optimizer.step()

        tokens_processed += tokens_per_opt_step

        # Console Progress & W&B Logging
        if step % log_interval == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            global_loss = reduce_tensor(accum_loss, average=True).item()
            t1 = time.time()
            dt = t1 - t0
            tok_per_sec = (log_interval * tokens_per_opt_step) / dt if step > 0 else tokens_per_opt_step / dt
            t0 = time.time()
            pct = (tokens_processed / total_tokens * 100) if total_tokens else 0.0

            print_rank0(
                f"step {step:5d}/{max_steps} | loss: {global_loss:.4f} | lr: {lr:.2e} | "
                f"grad_norm: {norm:.2f} | tok/s: {tok_per_sec:,.0f} | progress: {pct:.1f}%"
            )

            logger.log(
                {
                    "train/loss": global_loss,
                    "train/lr": lr,
                    "train/grad_norm": norm.item() if isinstance(norm, torch.Tensor) else norm,
                    "train/tok_per_sec": tok_per_sec,
                    "train/tokens_processed": tokens_processed,
                },
                step=step,
            )

        # Periodic Snapshot Checkpointing
        if step > 0 and step % save_interval == 0:
            ckpt_path = os.path.join(cfg.output_dir, f"ckpt_step_{step}.pt")
            save_checkpoint_rank0(
                {
                    "step": step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(config),
                    "runtime_meta": {
                        "size_tag": cfg.size_tag,
                        "tokenizer_type": tok_type,
                        "tokenizer_path": tok_path,
                    },
                },
                ckpt_path,
            )
            print_rank0(f"Saved periodic checkpoint to {ckpt_path}")

    print_rank0(f"\nTraining Complete! Best Validation Loss: {best_val_loss:.4f}")
    logger.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()