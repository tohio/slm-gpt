"""
dpo/train.py: Direct Preference Optimization execution engine for slm-gpt.
Features concatenated forward passes for high throughput, PyTorch 2.6+ safe
deserialization, support for both local files and Hugging Face datasets, and dynamic config reconstruction.
"""

import os
# Configure PyTorch virtual memory segments before any CUDA initialization
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.serialization
from torch.utils.data import DataLoader

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dpo.collator import DPODataCollator
from dpo.dataset import PreferenceDataset
from dpo.loss import DPOLoss, get_batch_logps
from tokenizer.factory import get_tokenizer
from transformer.config import ModelConfig
from transformer.model import DecoderOnlyTransformer

# Allowlist ModelConfig for PyTorch 2.6+ safe globals
try:
    torch.serialization.add_safe_globals([ModelConfig])
except AttributeError:
    pass

import warnings
warnings.filterwarnings("ignore", message=".*Argument aux_data.*cannot be converted to a JitArgument.*")


@dataclass
class DPOArgs:
    # Checkpoints & Data
    sft_model_path: str = "checkpoints/sft_125M/sft_final.pt"
    data_path: str = "data/dpo/preference_pairs.jsonl"
    output_dir: str = "checkpoints/dpo_125M"

    # Tokenizer
    tokenizer_type: str = "tiktoken"
    tokenizer_path: Optional[str] = None

    # Architecture fallback parameters
    vocab_size: int = 50304
    max_seq_len: int = 1024
    d_model: int = 768
    n_layers: int = 14
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ffn: int = 2048
    dropout: float = 0.0

    # Optimization Hyperparameters
    beta: float = 0.1
    learning_rate: float = 5e-7
    min_learning_rate: float = 5e-8
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch size = 32
    max_grad_norm: float = 1.0
    epochs: int = 1
    max_steps: Optional[int] = None
    warmup_ratio: float = 0.1
    log_interval_steps: int = 5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_cosine_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step > total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def _extract_turn_text(val: Any) -> str:
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        turns = []
        for turn in val:
            if isinstance(turn, dict):
                content = turn.get("content", "")
                turns.append(str(content).strip())
            else:
                turns.append(str(turn).strip())
        return "\n".join(turns).strip()
    return str(val).strip()


def normalize_preference_example(example: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Normalizes Hugging Face / JSONL records to standard prompt/chosen/rejected dicts."""
    prompt = example.get("prompt", "")
    chosen = example.get("chosen", "")
    rejected = example.get("rejected", "")

    # Multi-turn conversation format where chosen contains full message list
    if isinstance(chosen, list) and len(chosen) > 0 and isinstance(chosen[0], dict):
        if not prompt and len(chosen) >= 2:
            prompt_turns = [t.get("content", "") for t in chosen[:-1]]
            prompt = "\n".join(prompt_turns)
            chosen = chosen[-1].get("content", "")
        elif len(chosen) == 1:
            chosen = chosen[0].get("content", "")
        else:
            chosen = _extract_turn_text(chosen)

    if isinstance(rejected, list) and len(rejected) > 0 and isinstance(rejected[0], dict):
        if len(rejected) >= 2 and not prompt:
            rejected = rejected[-1].get("content", "")
        elif len(rejected) == 1:
            rejected = rejected[0].get("content", "")
        else:
            rejected = _extract_turn_text(rejected)

    prompt_str = _extract_turn_text(prompt)
    chosen_str = _extract_turn_text(chosen)
    rejected_str = _extract_turn_text(rejected)

    if not chosen_str or not rejected_str:
        return None

    return {
        "prompt": prompt_str,
        "chosen": chosen_str,
        "rejected": rejected_str,
    }


def resolve_model_config(checkpoint_path: str, args: DPOArgs, device: str) -> Tuple[ModelConfig, dict]:
    valid_fields = {f.name for f in fields(ModelConfig)}

    resolved_path = checkpoint_path
    if not os.path.exists(resolved_path):
        fallback = os.path.join(os.path.dirname(checkpoint_path), "sft_epoch_1.pt")
        if os.path.exists(fallback):
            print(f"[Notice] '{resolved_path}' not found. Falling back to '{fallback}'.")
            resolved_path = fallback

    if os.path.exists(resolved_path):
        print(f"[Loading] Restoring baseline weights from '{resolved_path}'...")
        ckpt = torch.load(resolved_path, map_location=device, weights_only=False)
        cfg_raw = ckpt.get("config", {})

        if isinstance(cfg_raw, ModelConfig):
            cfg = cfg_raw
        elif isinstance(cfg_raw, dict) and cfg_raw:
            filtered = {k: v for k, v in cfg_raw.items() if k in valid_fields}
            cfg = ModelConfig(**filtered)
        else:
            cfg = ModelConfig(
                vocab_size=args.vocab_size,
                max_seq_len=args.max_seq_len,
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                n_kv_heads=args.n_kv_heads,
                d_ffn=args.d_ffn,
                dropout=args.dropout,
                bias=False,
            )
        state_dict = ckpt.get("model_state_dict", ckpt)
        return cfg, state_dict

    print(f"[Warning] No checkpoint found. Initializing config from scratch.")
    cfg = ModelConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_ffn=args.d_ffn,
        dropout=args.dropout,
        bias=False,
    )
    return cfg, {}


def parse_args_from_cli(args_target) -> DPOArgs:
    """Loads default dataclass arguments and applies key=value CLI overrides safely."""
    cls = args_target if isinstance(args_target, type) else args_target.__class__
    instance = args_target if not isinstance(args_target, type) else args_target()

    int_fields = {
        "vocab_size", "max_seq_len", "d_model", "n_layers", "n_heads", "n_kv_heads", "d_ffn",
        "per_device_batch_size", "gradient_accumulation_steps", "epochs", "log_interval_steps",
        "seed", "max_steps"
    }
    float_fields = {"beta", "learning_rate", "min_learning_rate", "max_grad_norm", "warmup_ratio", "dropout"}

    kwargs = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            k = k.lstrip("-")
            if hasattr(instance, k):
                if k in int_fields:
                    kwargs[k] = int(v) if v.lower() != "none" else None
                elif k in float_fields:
                    kwargs[k] = float(v) if v.lower() != "none" else None
                elif isinstance(getattr(instance, k), bool):
                    kwargs[k] = v.lower() in ("true", "1", "yes")
                else:
                    kwargs[k] = v if v.lower() != "none" else None

    data = {f.name: getattr(instance, f.name) for f in fields(cls)}
    data.update(kwargs)
    return cls(**data)


def train_dpo(args: DPOArgs):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"--- Launching DPO Training Pipeline on {args.device.upper()} ---")

    dtype = torch.bfloat16 if (args.device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    use_autocast = (args.device == "cuda") and (dtype in (torch.float16, torch.bfloat16))

    # 1. Models Initialization
    config, state_dict = resolve_model_config(args.sft_model_path, args, args.device)
    config.dropout = 0.0

    policy_model = DecoderOnlyTransformer(config).to(args.device)
    ref_model = DecoderOnlyTransformer(config).to(args.device)

    if state_dict:
        policy_model.load_state_dict(state_dict, strict=False)
        ref_model.load_state_dict(state_dict, strict=False)

    policy_model.lm_head.weight = policy_model.wte.weight
    ref_model.lm_head.weight = ref_model.wte.weight

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # 2. Dataset & Collator
    tokenizer = get_tokenizer(args.tokenizer_type, args.tokenizer_path)
    pad_id = getattr(tokenizer, "pad_id", getattr(tokenizer, "pad_token_id", 50259))
    collator = DPODataCollator(pad_token_id=pad_id, pad_to_multiple_of=16)

    dataset = None
    if os.path.exists(args.data_path):
        print(f"[Loading] Loading local preference dataset from '{args.data_path}'...")
        try:
            dataset = PreferenceDataset(args.data_path, tokenizer=tokenizer, max_seq_len=config.max_seq_len)
        except Exception as e:
            print(f"[Warning] Failed to load local path directly: {e}")

    # Fallback to Hugging Face Hub if local file does not exist or failed
    if dataset is None or len(dataset) == 0:
        try:
            from datasets import load_dataset
            print(f"[DPO Data] Fetching dataset '{args.data_path}' from Hugging Face Hub...")
            raw_ds = load_dataset(args.data_path, split="train")

            formatted_data = []
            for ex in raw_ds:
                norm = normalize_preference_example(ex)
                if norm:
                    formatted_data.append(norm)

            print(f"[DPO Data] Successfully parsed {len(formatted_data):,} preference pairs from '{args.data_path}'.")
            dataset = PreferenceDataset(formatted_data, tokenizer=tokenizer, max_seq_len=config.max_seq_len)
        except Exception as e:
            print(f"[Notice] Could not load '{args.data_path}' from Hugging Face Hub ({e}).")
            print("Initializing verification batch of 64 mock pairs.")
            mock_data = [
                {
                    "prompt": f"Question {i}: What is the capital of France?",
                    "chosen": "The capital of France is Paris.",
                    "rejected": "The capital of France is London and it has many people living in it.",
                }
                for i in range(64)
            ]
            dataset = PreferenceDataset(mock_data, tokenizer=tokenizer, max_seq_len=config.max_seq_len)

    print(f"[Dataset] Active preference dataset contains {len(dataset):,} samples.")
    if len(dataset) < 100:
        print("⚠️ Warning: Preference dataset is very small. Ensure production runs use a full dataset.")

    drop_last = len(dataset) >= args.per_device_batch_size
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=(args.device == "cuda"),
        drop_last=drop_last,
    )

    # 3. Optimizer & Objective
    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=(args.device == "cuda"),
    )
    dpo_criterion = DPOLoss(beta=args.beta)

    steps_per_epoch = max(1, math.ceil(len(loader) / max(1, args.gradient_accumulation_steps)))
    total_steps = max(1, steps_per_epoch * args.epochs)
    if args.max_steps is not None and args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    warmup_steps = int(total_steps * args.warmup_ratio)
    global_step = 0

    policy_model.train()
    optimizer.zero_grad(set_to_none=True)

    print(f"Total DPO Optimization Steps: {total_steps} (Warmup: {warmup_steps})")
    print("-" * 65)

    for epoch in range(args.epochs):
        for step_idx, batch in enumerate(loader):
            c_ids = batch["chosen_input_ids"].to(args.device, non_blocking=True)
            c_labels = batch["chosen_labels"].to(args.device, non_blocking=True)
            r_ids = batch["rejected_input_ids"].to(args.device, non_blocking=True)
            r_labels = batch["rejected_labels"].to(args.device, non_blocking=True)

            batch_size = c_ids.size(0)
            all_ids = torch.cat([c_ids, r_ids], dim=0)

            with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
                with torch.no_grad():
                    all_ref_logits, _, _ = ref_model(all_ids)
                    ref_c_logits, ref_r_logits = all_ref_logits.split(batch_size, dim=0)
                    ref_c_logps = get_batch_logps(ref_c_logits, c_labels)
                    ref_r_logps = get_batch_logps(ref_r_logits, r_labels)

                all_pi_logits, _, _ = policy_model(all_ids)
                pi_c_logits, pi_r_logits = all_pi_logits.split(batch_size, dim=0)
                pi_c_logps = get_batch_logps(pi_c_logits, c_labels)
                pi_r_logps = get_batch_logps(pi_r_logits, r_labels)

                loss, metrics = dpo_criterion(pi_c_logps, pi_r_logps, ref_c_logps, ref_r_logps)
                loss_scaled = loss / args.gradient_accumulation_steps

            loss_scaled.backward()

            if (step_idx + 1) % args.gradient_accumulation_steps == 0 or (step_idx + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.max_grad_norm)

                lr = get_cosine_lr(global_step, total_steps, warmup_steps, args.learning_rate, args.min_learning_rate)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_interval_steps == 0 or global_step == total_steps:
                    print(
                        f"Step {global_step:04d}/{total_steps:04d} | "
                        f"LR: {lr:.2e} | "
                        f"Loss: {metrics['loss'].item():.4f} | "
                        f"Acc: {metrics['accuracy'].item() * 100:.1f}% | "
                        f"Margin: {metrics['reward_margin'].item():.3f} | "
                        f"Chosen R: {metrics['chosen_rewards'].item():.3f} | "
                        f"Rejected R: {metrics['rejected_rewards'].item():.3f}"
                    )

                if global_step >= total_steps:
                    break

        if global_step >= total_steps:
            break

    save_path = os.path.join(args.output_dir, "dpo_final.pt")
    torch.save(
        {
            "global_step": global_step,
            "model_state_dict": policy_model.state_dict(),
            "config": asdict(config) if hasattr(config, "__dict__") else config,
        },
        save_path,
    )
    print(f"✓ DPO Training Completed. Aligned policy saved to: {save_path}")


if __name__ == "__main__":
    train_dpo(parse_args_from_cli(DPOArgs))