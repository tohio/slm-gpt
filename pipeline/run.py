"""
pipeline/run.py: Autonomous End-to-End Pipeline Orchestrator for slm-gpt.
Integrates HardwareProfiler, Stage 0 Data Ingestion & Packing, and 
sequential Pre-training -> SFT -> DPO execution with Health Gates.
"""

from dataclasses import dataclass
import glob
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import torch

from scaling.config import RuntimeConfig
from scaling.profiler import HardwareProfiler, tune_micro_batch_size
from tokenizer.factory import get_tokenizer


@dataclass
class PipelineConfig:
    size: str = "125M"
    tokenizer_type: str = "tiktoken"
    tokenizer_path: Optional[str] = None
    num_gpus: Optional[int] = None
    resume_from: Optional[str] = None  # None, "sft", or "dpo"

    # Stage Data Paths
    pretrain_data_dir: str = "data/pretrain"
    sft_data_path: str = "data/sft/train.jsonl"
    dpo_data_path: str = "data/dpo/preference_pairs.jsonl"

    # Stage 0 Data Budgets (Used if data is missing on cold-start)
    bootstrap_pretrain_tokens: int = 10_000_000
    bootstrap_sft_samples: int = 5_000
    bootstrap_dpo_samples: int = 2_000

    # Training Budgets
    pretrain_max_steps: Optional[int] = None
    sft_epochs: int = 3
    dpo_epochs: int = 1


def parse_cli_args(args_cls: type[PipelineConfig]) -> PipelineConfig:
    kwargs: Dict[str, Any] = {}
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


class PipelineOrchestrator:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

        # 1. Profile Hardware & Auto-Generate Environment Requirements
        print("=" * 70)
        print("  PROFILING LOCAL HARDWARE & ENVIRONMENT")
        print("=" * 70)
        self.hw_profile = HardwareProfiler.profile()
        HardwareProfiler.generate_requirements_file("requirements.txt")
        print(f"✓ Detected Device: {self.hw_profile.device_name} ({self.hw_profile.arch_generation})")
        print(f"✓ VRAM / Memory:   {self.hw_profile.total_memory_gb:.1f} GB")
        print(f"✓ Precision:       {self.hw_profile.precision_str}")
        print("=" * 70)

        # 2. Resolve Available Compute
        detected_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.num_gpus = cfg.num_gpus if cfg.num_gpus is not None else max(1, detected_gpus)

        # 3. Derive Dynamic Runtime Parameters
        tok = get_tokenizer(cfg.tokenizer_type, cfg.tokenizer_path)
        runtime_cfg = RuntimeConfig.build(
            size=cfg.size,
            vocab_size=tok.vocab_size,
            tokenizer_type=cfg.tokenizer_type,
            tokenizer_path=cfg.tokenizer_path,
        )
        self.size_tag = runtime_cfg.size_tag

        # Auto-tune micro batch size based on actual profiled VRAM
        self.auto_micro_batch = tune_micro_batch_size(
            target_params=runtime_cfg.actual_params,
            max_seq_len=runtime_cfg.model_cfg.max_seq_len,
            total_memory_gb=self.hw_profile.total_memory_gb,
            world_size=self.num_gpus,
        )
        print(f"[Auto-Tuner] Recommended Micro-Batch Size per GPU: {self.auto_micro_batch}")

        # 4. Dynamic Checkpoint Paths
        self.pretrain_ckpt = f"checkpoints/pretrain_{self.size_tag}/best_model.pt"
        self.sft_ckpt = f"checkpoints/sft_{self.size_tag}/sft_final.pt"
        self.dpo_ckpt = f"checkpoints/dpo_{self.size_tag}/dpo_final.pt"

    def print_stage_banner(self, name: str):
        print("\n" + "=" * 70)
        print(f"  PIPELINE STAGE: {name.upper()}")
        print("=" * 70)

    def verify_checkpoint_health(self, path: str, min_size_mb: float = 1.0) -> bool:
        if not os.path.exists(path):
            print(f"❌ Health Gate Error: File not found at '{path}'")
            return False
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb < min_size_mb:
            print(f"❌ Health Gate Error: File size too small ({size_mb:.2f} MB) at '{path}'")
            return False
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if "model_state_dict" not in ckpt and not isinstance(ckpt, dict):
                print(f"❌ Health Gate Error: Invalid checkpoint structure at '{path}'")
                return False
            return True
        except Exception as e:
            print(f"❌ Health Gate Error: Failed to deserialize '{path}': {e}")
            return False

    def ensure_data_prepared(self):
        """Stage 0: Autonomous Cold-Start Data Ingestion & Packing Gate."""
        self.print_stage_banner("0. Autonomous Data Ingestion & Verification")

        # 0A. Pre-training Shard Check
        has_train = bool(
            os.path.exists(os.path.join(self.cfg.pretrain_data_dir, "train.bin"))
            or glob.glob(os.path.join(self.cfg.pretrain_data_dir, "train_*.bin"))
        )
        has_val = bool(
            os.path.exists(os.path.join(self.cfg.pretrain_data_dir, "val.bin"))
            or glob.glob(os.path.join(self.cfg.pretrain_data_dir, "val_*.bin"))
        )

        if not (has_train and has_val) and self.cfg.resume_from not in ("sft", "dpo"):
            print(f"⚠️ Pre-training binary shards missing in '{self.cfg.pretrain_data_dir}'.")
            print(f"   Executing token packing curriculum target: {self.cfg.bootstrap_pretrain_tokens:,} tokens...")
            cmd = [
                sys.executable,
                "-m", "data.prepare_packed_curriculum",
                f"output_dir={self.cfg.pretrain_data_dir}",
                f"total_tokens={self.cfg.bootstrap_pretrain_tokens}",
            ]
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                raise RuntimeError("Stage 0: Pre-training token packing failed.")
        else:
            print(f"✓ Pre-training shards verified in '{self.cfg.pretrain_data_dir}'")

        # 0B. SFT Dialogue Check
        if not os.path.exists(self.cfg.sft_data_path) and self.cfg.resume_from != "dpo":
            print(f"⚠️ SFT dataset missing at '{self.cfg.sft_data_path}'. Ingesting SmolTalk...")
            out_dir = os.path.dirname(self.cfg.sft_data_path) or "data/sft"
            cmd = [
                sys.executable,
                "-m", "data.prepare_sft",
                "--source=HuggingFaceTB/smoltalk",
                f"--output_dir={out_dir}",
                f"--max_samples={self.cfg.bootstrap_sft_samples}",
            ]
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                raise RuntimeError("Stage 0: SFT dataset preparation failed.")
        else:
            print(f"✓ SFT dataset verified at '{self.cfg.sft_data_path}'")

        # 0C. DPO Preference Check
        if not os.path.exists(self.cfg.dpo_data_path):
            print(f"⚠️ DPO dataset missing at '{self.cfg.dpo_data_path}'. Extracting preference pairs...")
            cmd = [
                sys.executable,
                "-m", "data.prepare_dpo",
                f"--output_path={self.cfg.dpo_data_path}",
                f"--max_samples={self.cfg.bootstrap_dpo_samples}",
            ]
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                raise RuntimeError("Stage 0: DPO dataset preparation failed.")
        else:
            print(f"✓ DPO dataset verified at '{self.cfg.dpo_data_path}'")

    def run_pretrain(self):
        self.print_stage_banner("1. Pre-training (Multi-GPU DDP)")
        if self.cfg.resume_from in ("sft", "dpo"):
            print(f"⏩ Skipping Pre-training (resuming from stage '{self.cfg.resume_from}').")
            return

        cmd = [
            "torchrun",
            f"--nproc_per_node={self.num_gpus}",
            "-m", "pretrain.train",
            f"size={self.cfg.size}",
            f"data_dir={self.cfg.pretrain_data_dir}",
            f"tokenizer_type={self.cfg.tokenizer_type}",
            f"micro_batch_size={self.auto_micro_batch}",
        ]
        if self.cfg.tokenizer_path:
            cmd.append(f"tokenizer_path={self.cfg.tokenizer_path}")
        if self.cfg.pretrain_max_steps:
            cmd.append(f"max_steps={self.cfg.pretrain_max_steps}")

        print(f"Command: {' '.join(cmd)}")
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            raise RuntimeError(f"Pre-training failed with exit code {ret.returncode}")

        if not self.verify_checkpoint_health(self.pretrain_ckpt):
            raise RuntimeError(f"Pre-training finished but failed health gate: '{self.pretrain_ckpt}'")
        print(f"✓ Health Gate Passed for Pre-training: {self.pretrain_ckpt}")

    def run_sft(self):
        self.print_stage_banner("2. Supervised Fine-Tuning (Single-GPU)")
        if self.cfg.resume_from == "dpo":
            print(f"⏩ Skipping SFT (resuming from stage '{self.cfg.resume_from}').")
            return

        if not os.path.exists(self.pretrain_ckpt):
            raise FileNotFoundError(f"Base checkpoint required for SFT not found: '{self.pretrain_ckpt}'")

        cmd = [
            sys.executable,
            "-m", "sft.train",
            f"pretrained_ckpt={self.pretrain_ckpt}",
            f"output_dir=checkpoints/sft_{self.size_tag}",
            f"data_path={self.cfg.sft_data_path}",
            f"epochs={self.cfg.sft_epochs}",
            f"tokenizer_type={self.cfg.tokenizer_type}",
            f"per_device_batch_size={self.auto_micro_batch}",
        ]
        if self.cfg.tokenizer_path:
            cmd.append(f"tokenizer_path={self.cfg.tokenizer_path}")

        print(f"Command: {' '.join(cmd)}")
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            raise RuntimeError(f"SFT failed with exit code {ret.returncode}")

        if not self.verify_checkpoint_health(self.sft_ckpt):
            raise RuntimeError(f"SFT finished but failed health gate: '{self.sft_ckpt}'")
        print(f"✓ Health Gate Passed for SFT: {self.sft_ckpt}")

    def run_dpo(self):
        self.print_stage_banner("3. Direct Preference Optimization (Single-GPU)")
        if not os.path.exists(self.sft_ckpt):
            raise FileNotFoundError(f"Reference SFT checkpoint required for DPO not found: '{self.sft_ckpt}'")

        cmd = [
            sys.executable,
            "-m", "dpo.train",
            f"sft_model_path={self.sft_ckpt}",
            f"output_dir=checkpoints/dpo_{self.size_tag}",
            f"data_path={self.cfg.dpo_data_path}",
            f"epochs={self.cfg.dpo_epochs}",
            f"tokenizer_type={self.cfg.tokenizer_type}",
            f"per_device_batch_size={max(1, self.auto_micro_batch // 2)}",
        ]
        if self.cfg.tokenizer_path:
            cmd.append(f"tokenizer_path={self.cfg.tokenizer_path}")

        print(f"Command: {' '.join(cmd)}")
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            raise RuntimeError(f"DPO failed with exit code {ret.returncode}")

        if not self.verify_checkpoint_health(self.dpo_ckpt):
            raise RuntimeError(f"DPO finished but failed health gate: '{self.dpo_ckpt}'")
        print(f"✓ Health Gate Passed for DPO: {self.dpo_ckpt}")

    def execute(self):
        start_time = time.time()
        print("=" * 70)
        print("       slm-gpt Autonomous Production Pipeline Orchestrator        ")
        print("=" * 70)
        print(f"Model Size Target:      {self.cfg.size} (Tag: {self.size_tag})")
        print(f"Hardware Profile:       {self.hw_profile.device_name} ({self.hw_profile.arch_generation})")
        print(f"Pre-training Compute:   Multi-GPU ({self.num_gpus} GPUs via torchrun)")
        print(f"Alignment Compute:      Single-GPU (SFT & DPO)")
        print("=" * 70)

        # Stage 0: Data verification & bootstrap
        self.ensure_data_prepared()

        # Stages 1-3: Core training pipeline
        self.run_pretrain()
        self.run_sft()
        self.run_dpo()

        elapsed_mins = (time.time() - start_time) / 60.0
        print("\n" + "=" * 70)
        print(f"✓ Pipeline Run Successfully Completed in {elapsed_mins:.2f} minutes.")
        print(f"Final Aligned Model: {self.dpo_ckpt}")
        print("=" * 70)


def main():
    cfg = parse_cli_args(PipelineConfig)
    orchestrator = PipelineOrchestrator(cfg)
    orchestrator.execute()


if __name__ == "__main__":
    main()