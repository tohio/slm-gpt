"""
scaling/config.py: Typed runtime configuration object for slm-gpt.
Eliminates static YAML configs and eliminates argparse boilerplate.
"""

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Dict, Optional, Union

import torch

from scaling.engine import (
    derive_learning_rate,
    derive_model_config,
    derive_training_budget,
    format_size,
    parse_size_str,
)
from scaling.hardware import HardwareProfile, profile_hardware
from transformer.config import ModelConfig


@dataclass
class RuntimeConfig:
    # Target parameter specification
    size: str = "125M"
    target_params: int = 125_000_000
    actual_params: int = 0
    size_tag: str = ""

    # Pipeline Stage & Directory Resolution
    stage: str = "pretrain"  # 'pretrain', 'sft', 'dpo'
    output_dir: str = ""

    # Resolved Architecture
    model_cfg: Optional[ModelConfig] = None
    max_seq_len: int = 2048

    # Scaling Law Token & Step Budgets (Default: 50:1 tokens per parameter)
    tokens_per_param: float = 50.0
    target_tokens: int = 0
    total_steps: int = 0
    warmup_steps: int = 0

    # Batching & Optimization
    global_batch_size: int = 64
    micro_batch_size: int = 4
    grad_accum_steps: int = 1
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Hardware Environment
    hardware: Optional[HardwareProfile] = None

    # Observability & Tracking
    wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Tokenizer Configuration
    tokenizer_type: str = "tiktoken"
    tokenizer_path: Optional[str] = None

    @classmethod
    def build(
        cls,
        size: Union[str, int] = "125M",
        vocab_size: int = 50257,
        stage: str = "pretrain",
        max_seq_len: int = 2048,
        global_batch_size: int = 64,
        micro_batch_size: Optional[int] = None,
        tokens_per_param: float = 50.0,
        tokenizer_type: str = "tiktoken",
        tokenizer_path: Optional[str] = None,
        world_size: int = 1,
        **custom_overrides,
    ) -> "RuntimeConfig":
        """
        Dynamically constructs a complete runtime configuration.
        """
        target_int = parse_size_str(size)

        # 1. Derive Architecture (enforces min_layers=8)
        model_cfg, actual_params = derive_model_config(
            target_params=target_int,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )

        size_tag = format_size(actual_params)

        # 2. Derive Learning Rates
        max_lr, min_lr = derive_learning_rate(model_cfg.d_model)

        # 3. Derive Scaling Law Token & Step Budgets (50:1)
        target_tokens, total_steps, warmup_steps = derive_training_budget(
            param_count=actual_params,
            tokens_per_param=tokens_per_param,
            max_seq_len=max_seq_len,
            global_batch_size=global_batch_size,
        )

        # 4. Profile Hardware
        hw = profile_hardware(max_seq_len=max_seq_len, d_model=model_cfg.d_model)
        resolved_micro_batch = micro_batch_size if micro_batch_size is not None else hw.recommended_micro_batch

        # 5. Calculate Gradient Accumulation Steps across the cluster
        per_step_capacity = resolved_micro_batch * world_size
        grad_accum = max(1, global_batch_size // per_step_capacity)

        # 6. Dynamic Output Directory (zero hardcoded size paths)
        output_dir = f"checkpoints/{stage}_{size_tag}"

        instance = cls(
            size=str(size),
            target_params=target_int,
            actual_params=actual_params,
            size_tag=size_tag,
            stage=stage,
            output_dir=output_dir,
            model_cfg=model_cfg,
            max_seq_len=max_seq_len,
            tokens_per_param=tokens_per_param,
            target_tokens=target_tokens,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            global_batch_size=global_batch_size,
            micro_batch_size=resolved_micro_batch,
            grad_accum_steps=grad_accum,
            max_lr=max_lr,
            min_lr=min_lr,
            hardware=hw,
            tokenizer_type=tokenizer_type,
            tokenizer_path=tokenizer_path,
        )

        # Apply any explicit manual overrides
        for k, v in custom_overrides.items():
            if hasattr(instance, k):
                setattr(instance, k, v)

        return instance

    def save_json(self, filepath: str):
        """Serializes configuration for artifact tracking."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = asdict(self)
        if self.model_cfg:
            data["model_cfg"] = asdict(self.model_cfg)
        if self.hardware:
            data["hardware"] = {
                "device_name": self.hardware.device_name,
                "vram_gb": self.hardware.vram_gb,
                "precision": self.hardware.precision_str,
            }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)