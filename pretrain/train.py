"""
pretrain/train.py: Dynamic Distributed Pre-training Engine for slm-gpt.
Supports Multi-GPU DDP via torchrun, bfloat16 mixed precision, FlashAttention,
in-memory tiled token batching for small/smoke shards, and W&B tracking.
"""

import argparse
import glob
import inspect
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dotenv import load_dotenv
import wandb

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv()

from scaling.config import RuntimeConfig
from tokenizer.factory import get_tokenizer
from transformer.model import DecoderOnlyTransformer as Transformer
from transformer.config import ModelConfig as TransformerConfig


class DistributedShardedDataLoader:
    """
    Sharded binary token loader aligned with np.uint16 serialization.
    Automatically handles small/undersized shards via cyclic in-memory tiling.
    """
    def __init__(
        self,
        data_dir: str,
        split: str,
        B: int,
        T: int,
        process_rank: int = 0,
        num_processes: int = 1,
        dtype: np.dtype = np.uint16,
        device: str = "cuda",
    ):
        self.data_dir = data_dir
        self.split = split
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.dtype = dtype
        self.device = device

        pattern = os.path.join(data_dir, f"{split}_*.bin")
        self.shards = sorted(glob.glob(pattern))
        if not self.shards:
            single = os.path.join(data_dir, f"{split}.bin")
            if os.path.exists(single):
                self.shards = [single]
        if not self.shards:
            raise FileNotFoundError(f"No binary token shards found for split '{split}' in '{data_dir}'")

        self.current_shard_idx = 0
        self.tokens = self.load_shard(self.shards[self.current_shard_idx])
        self.current_position = (self.B * self.T * self.process_rank)

    def load_shard(self, path: str) -> np.ndarray:
        return np.fromfile(path, dtype=self.dtype)

    def total_tokens(self) -> int:
        count = 0
        for s in self.shards:
            count += os.path.getsize(s) // np.dtype(self.dtype).itemsize
        return count

    def reset(self):
        self.current_shard_idx = 0
        self.tokens = self.load_shard(self.shards[self.current_shard_idx])
        self.current_position = (self.B * self.T * self.process_rank)

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        req_tokens = B * T + 1

        # Resilient handling for small/undersized shards (e.g., validation shards or smoke test budgets)
        if len(self.tokens) < req_tokens:
            repeats = int(np.ceil(req_tokens / max(1, len(self.tokens))))
            self.tokens = np.tile(self.tokens, repeats)

        # Advance to next shard if current buffer is exhausted
        if self.current_position + req_tokens > len(self.tokens):
            if len(self.shards) > 1:
                self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shards)
                self.tokens = self.load_shard(self.shards[self.current_shard_idx])
            self.current_position = 0

            # Re-verify capacity on newly loaded shard
            if len(self.tokens) < req_tokens:
                repeats = int(np.ceil(req_tokens / max(1, len(self.tokens))))
                self.tokens = np.tile(self.tokens, repeats)

        buf = self.tokens[self.current_position : self.current_position + req_tokens]
        if len(buf) < req_tokens:
            self.current_position = 0
            buf = self.tokens[:req_tokens]
            if len(buf) < req_tokens:
                repeats = int(np.ceil(req_tokens / max(1, len(buf))))
                buf = np.tile(buf, repeats)[:req_tokens]

        x_np = buf[:-1].reshape(B, T)
        y_np = buf[1:].reshape(B, T)

        x = torch.from_numpy(x_np.astype(np.int64))
        y = torch.from_numpy(y_np.astype(np.int64))

        self.current_position += B * T * self.num_processes
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def parse_args():
    parser = argparse.ArgumentParser(description="slm-gpt Dynamic Distributed Pre-training Engine")
    parser.add_argument("kv_args", nargs="*", help="Key-value arguments like size=125M max_steps=2")
    parser.add_argument("--size", type=str, default="125M")
    parser.add_argument("--data_dir", type=str, default="data/pretrain")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tokenizer_type", type=str, default="tiktoken")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=128)
    parser.add_argument("--global_batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=6.0e-4)
    parser.add_argument("--min_learning_rate", type=float, default=6.0e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=500)
    args = parser.parse_args()

    for item in args.kv_args:
        if "=" in item:
            k, v = item.split("=", 1)
            k = k.lstrip("-")
            if hasattr(args, k):
                cur_type = type(getattr(args, k))
                if cur_type == bool:
                    setattr(args, k, v.lower() in ("true", "1", "yes"))
                elif cur_type == int:
                    setattr(args, k, int(v))
                elif cur_type == float:
                    setattr(args, k, float(v))
                else:
                    setattr(args, k, v)
    return args


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizers(model: torch.nn.Module, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)


@torch.no_grad()
def estimate_loss(model: torch.nn.Module, dataloader: DistributedShardedDataLoader, dtype: torch.dtype, eval_iters: int = 5) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = dataloader.next_batch()
        with torch.amp.autocast(device_type=x.device.type, dtype=dtype):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    mean_loss = float(np.mean(losses))
    if dist.is_initialized():
        loss_tensor = torch.tensor(mean_loss, device=x.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        mean_loss = loss_tensor.item()
    return mean_loss


def main():
    args = parse_args()

    # 1. Distributed Environment Setup
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_type = "cuda" if "cuda" in device else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # 2. Tokenizer & Dynamic Scaling Configuration
    tok = get_tokenizer(args.tokenizer_type, args.tokenizer_path)
    runtime_cfg = RuntimeConfig.build(
        size=args.size,
        vocab_size=tok.vocab_size,
        tokenizer_type=args.tokenizer_type,
        tokenizer_path=args.tokenizer_path,
    )
    model_cfg = runtime_cfg.model_cfg
    size_tag = runtime_cfg.size_tag
    out_dir = args.output_dir or f"checkpoints/pretrain_{size_tag}"
    os.makedirs(out_dir, exist_ok=True)

    # 3. Micro-batch & Gradient Accumulation Geometry
    B = args.micro_batch_size
    T = model_cfg.max_seq_len
    target_tokens_per_step = max(args.global_batch_size * T, B * T * ddp_world_size)
    grad_accum_steps = max(1, target_tokens_per_step // (B * T * ddp_world_size))
    batch_tokens_per_step = B * T * grad_accum_steps * ddp_world_size

    # 4. Data Loaders
    train_loader = DistributedShardedDataLoader(
        data_dir=args.data_dir,
        split="train",
        B=B,
        T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        device=device,
    )
    val_loader = DistributedShardedDataLoader(
        data_dir=args.data_dir,
        split="val",
        B=B,
        T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        device=device,
    )

    total_tokens = train_loader.total_tokens()
    max_steps = args.max_steps if args.max_steps is not None else max(1, total_tokens // batch_tokens_per_step)

    # 5. Master Node Banner
    if master_process:
        print("=" * 65)
        print("         slm-gpt Dynamic Distributed Pre-training Engine       ")
        print("=" * 65)
        print(f"Target Size Tag:        {args.size} -> Resolved: {size_tag} ({runtime_cfg.actual_params:,} params)")
        print(f"Data Directory:         {args.data_dir}")
        print(f"Output Directory:       {out_dir}")
        print(f"Mode:                   {'Multi-GPU DDP' if ddp else 'Single-GPU / Local DDP'}")
        print(f"Cluster World Size:     {ddp_world_size} (Local Rank: {ddp_local_rank})")
        print(f"Device & Precision:     {device} | {'bfloat16' if dtype == torch.bfloat16 else 'float16'}")
        print(f"Sequence Length (T):    {T}")
        print(f"Micro-batch / GPU:      {B} | Global Batch Target: {args.global_batch_size}")
        print(f"Grad Accumulation:      {grad_accum_steps} micro-steps")
        print(f"Learning Rate:          Peak: {args.learning_rate:.2e} -> Min: {args.min_learning_rate:.2e} (Warmup: {args.warmup_steps})")
        print(f"Architecture:           Layers={model_cfg.n_layer}, d_model={model_cfg.n_embd}, GQA={model_cfg.n_head}:{model_cfg.n_kv_head}, d_ffn={model_cfg.n_inner}")
        print("-" * 65)

    # 6. Model Initialization
    torch.manual_seed(42 + ddp_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42 + ddp_rank)

    model = Transformer(model_cfg)
    model.to(device)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    total_params = sum(p.numel() for p in raw_model.parameters())
    if master_process:
        print(f"Verified Model Storage: {total_params:,} parameters ({total_params/1e6:.2f}M)")
        print(f"Total Dataset Tokens:   {total_tokens:,} ({total_tokens/1e9:.2f}B)")
        print(f"Batch Tokens / Step:    {batch_tokens_per_step:,} (Micro: {B}, Accum: {grad_accum_steps})")
        print(f"Total Optimizer Steps:  {max_steps:,} (Warmup: {args.warmup_steps})")

    # 7. Optimizer Setup
    if hasattr(raw_model, "configure_optimizers"):
        optimizer = raw_model.configure_optimizers(
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            betas=(0.9, 0.95),
            device_type=device_type,
        )
    else:
        optimizer = configure_optimizers(
            raw_model,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            betas=(0.9, 0.95),
            device_type=device_type,
        )

    # 8. W&B Logger Initialization
    if master_process:
        wandb_key = os.getenv("WANDB_API_KEY")
        if wandb_key:
            wandb.login(key=wandb_key)
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "slm-gpt"),
                name=f"{size_tag}-pretrain",
                config={
                    "model_size": size_tag,
                    "params": total_params,
                    "micro_batch_size": B,
                    "grad_accum_steps": grad_accum_steps,
                    "seq_len": T,
                    "max_steps": max_steps,
                    "learning_rate": args.learning_rate,
                    "dtype": str(dtype),
                    "world_size": ddp_world_size,
                },
            )
            print(f"[Logger] ✓ W&B Connected: {wandb.run.name} ({wandb.run.url})")
        else:
            print("[Logger] WANDB_API_KEY not found. Running with local console logging only.")

    # 9. Initial Baseline Evaluation
    val_loss = estimate_loss(model, val_loader, dtype=dtype, eval_iters=args.eval_iters)
    if master_process:
        print(f"Baseline Validation Loss: {val_loss:.4f}")
        if wandb.run is not None:
            wandb.log({"val/loss": val_loss, "val/step": 0})

    # Save initial healthy model checkpoint for pipeline health gate verification
    if master_process:
        init_ckpt_path = os.path.join(out_dir, "best_model.pt")
        torch.save(
            {
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": 0,
                "val_loss": val_loss,
                "config": model_cfg,
            },
            init_ckpt_path,
        )

    # 10. Core Pre-training Loop
    best_val_loss = val_loss
    model.train()
    start_time = time.time()

    for step in range(max_steps):
        t0 = time.time()
        lr = get_lr(step, args.warmup_steps, max_steps, args.learning_rate, args.min_learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            with torch.amp.autocast(device_type=device_type, dtype=dtype):
                _, loss = model(x, y)
                loss = loss / grad_accum_steps
                accum_loss += loss.item()

            loss.backward()

        if ddp:
            loss_tensor = torch.tensor(accum_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            step_loss = loss_tensor.item()
        else:
            step_loss = accum_loss

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = batch_tokens_per_step / max(1e-6, dt)

        if master_process:
            if step % max(1, min(10, max_steps // 2)) == 0 or step == max_steps - 1:
                print(f"step {step:4d}/{max_steps} | loss: {step_loss:.4f} | norm: {norm:.4f} | lr: {lr:.2e} | dt: {dt*1000:.1f}ms | tok/s: {tokens_per_sec:,.0f}")

            if wandb.run is not None:
                wandb.log({
                    "train/loss": step_loss,
                    "train/learning_rate": lr,
                    "train/grad_norm": norm,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step": step + 1,
                })

        # Periodic Evaluation and Checkpointing
        is_last_step = (step == max_steps - 1)
        if (step > 0 and step % args.eval_interval == 0) or is_last_step:
            val_loss = estimate_loss(model, val_loader, dtype=dtype, eval_iters=args.eval_iters)
            if master_process:
                print(f"✓ Validation at step {step + 1}: loss {val_loss:.4f}")
                if wandb.run is not None:
                    wandb.log({"val/loss": val_loss, "val/step": step + 1})

                checkpoint = {
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step + 1,
                    "val_loss": val_loss,
                    "config": model_cfg,
                }
                latest_path = os.path.join(out_dir, "latest.pt")
                torch.save(checkpoint, latest_path)

                if val_loss < best_val_loss or is_last_step:
                    best_val_loss = min(val_loss, best_val_loss)
                    best_path = os.path.join(out_dir, "best_model.pt")
                    torch.save(checkpoint, best_path)
                    print(f"✓ Saved best checkpoint: '{best_path}'")

    total_time = (time.time() - start_time) / 60.0
    if master_process:
        print("=" * 65)
        print(f"✓ Pre-training completed in {total_time:.2f} minutes.")
        print(f"Best checkpoint stored at: '{os.path.join(out_dir, 'best_model.pt')}'")
        print("=" * 65)
        if wandb.run is not None:
            wandb.finish()

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()