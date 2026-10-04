"""
pipeline/run.py: Autonomous End-to-End Pipeline Orchestrator for slm-gpt.
Integrates HardwareProfiler, Stage 0 Data Ingestion & Packing, and 
sequential Pre-training -> SFT -> DPO execution with Health Gates.
"""

import os
# Configure PyTorch virtual memory segments for orchestrator and subprocesses
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Load local environment variables (.env) for API keys, tokens, and telemetry
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from dataclasses import dataclass
import glob
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Union

import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scaling.config import RuntimeConfig
from scaling.profiler import HardwareProfiler, tune_micro_batch_size
from tokenizer.factory import get_tokenizer


def parse_token_string(val: Optional[Union[str, int, float]]) -> Optional[int]:
    """
    Parses human-readable token strings like '5B', '14B', '500M', '500k', 
    or raw integers into exact integer token counts.
    """
    if val is None or str(val).strip().lower() in ("none", ""):
        return None
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip().replace("_", "").upper()
    multipliers = {
        "K": 1_000,
        "M": 1_000_000,
        "B": 1_000_000_000,
        "T": 1_000_000_000_000,
    }
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            try:
                return int(float(s[:-len(suffix)]) * mult)
            except ValueError:
                break
    try:
        return int(float(s))
    except ValueError:
        raise ValueError(f"Unable to parse token budget string: '{val}'")


@dataclass
class PipelineConfig:
    size: str = "125M"
    stage: Optional[str] = None             # None (all), "pretrain", "sft", "dpo", "prepare"
    tokens: Optional[str] = None            # e.g. "5B", "14B", "500M"
    tokenizer_type: str = "tiktoken"
    tokenizer_path: Optional[str] = None
    num_gpus: Optional[int] = None
    micro_batch_size: Optional[int] = None  # Explicit CLI override support
    global_batch_size: Optional[int] = None
    max_steps: Optional[int] = None         # CLI alias for pretrain_max_steps
    resume_from: Optional[str] = None       # None, "sft", or "dpo"
    output_dir: Optional[str] = None

    # Stage Data Paths
    pretrain_data_dir: str = "data/pretrain"
    sft_data_path: str = "data/sft/train.jsonl"
    dpo_data_path: str = "data/dpo/preference_pairs.jsonl"

    # Stage 0 Data Budgets (Used if data is missing on cold-start)
    bootstrap_pretrain_tokens: int = 10_000_000
    bootstrap_sft_samples: int = 5_000
    bootstrap_dpo_samples: int = 7_000

    # Training Budgets
    pretrain_max_steps: Optional[int] = None
    sft_epochs: int = 3
    dpo_epochs: int = 1
    dpo_max_steps: Optional[int] = None
    dpo_learning_rate: float = 5e-7


def parse_cli_args(args_cls: type[PipelineConfig]) -> PipelineConfig:
    kwargs: Dict[str, Any] = {}
    int_fields = {
        "num_gpus", "micro_batch_size", "global_batch_size", "bootstrap_pretrain_tokens",
        "bootstrap_sft_samples", "bootstrap_dpo_samples",
        "pretrain_max_steps", "max_steps", "sft_epochs", "dpo_epochs", "dpo_max_steps"
    }
    float_fields = {"dpo_learning_rate"}

    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            k = k.lstrip("-")
            if hasattr(args_cls, k):
                orig_val = getattr(args_cls, k)
                if isinstance(orig_val, bool):
                    kwargs[k] = v.lower() in ("true", "1", "yes")
                elif k in int_fields:
                    kwargs[k] = int(v) if v.lower() != "none" else None
                elif k in float_fields:
                    kwargs[k] = float(v) if v.lower() != "none" else None
                else:
                    kwargs[k] = v if v.lower() != "none" else None
    return args_cls(**kwargs)


class PipelineOrchestrator:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

        # 1. Defensive Subdirectory Resolution for Pre-training Data
        pretrain_path = Path(self.cfg.pretrain_data_dir)
        if (pretrain_path / "pretrain").is_dir() and not list(pretrain_path.glob("train_*.bin")) and not (pretrain_path / "train.bin").exists():
            self.cfg.pretrain_data_dir = str(pretrain_path / "pretrain")

        # 2. Profile Hardware & Auto-Generate Environment Requirements
        print("=" * 70)
        print("  PROFILING LOCAL HARDWARE & ENVIRONMENT")
        print("=" * 70)
        self.hw_profile = HardwareProfiler.profile()
        HardwareProfiler.generate_requirements_file("requirements.txt")
        print(f"✓ Detected Device: {self.hw_profile.device_name} ({self.hw_profile.arch_generation})")
        print(f"✓ VRAM / Memory:   {self.hw_profile.total_memory_gb:.1f} GB")
        print(f"✓ Precision:       {self.hw_profile.precision_str}")
        print("=" * 70)

        # 3. Resolve Available Compute
        detected_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.num_gpus = cfg.num_gpus if cfg.num_gpus is not None else max(1, detected_gpus)

        # 4. Derive Dynamic Runtime Parameters
        tok = get_tokenizer(cfg.tokenizer_type, cfg.tokenizer_path)
        runtime_cfg = RuntimeConfig.build(
            size=cfg.size,
            vocab_size=tok.vocab_size,
            tokenizer_type=cfg.tokenizer_type,
            tokenizer_path=cfg.tokenizer_path,
        )
        self.nominal_tag = cfg.size or runtime_cfg.size_tag
        self.size_tag = runtime_cfg.size_tag

        # Honor manual CLI micro_batch_size if specified, otherwise auto-tune
        if cfg.micro_batch_size is not None:
            self.auto_micro_batch = cfg.micro_batch_size
            print(f"[Config] Manual Micro-Batch Size per GPU specified: {self.auto_micro_batch}")
        else:
            self.auto_micro_batch = tune_micro_batch_size(
                target_params=runtime_cfg.actual_params,
                max_seq_len=runtime_cfg.model_cfg.max_seq_len,
                total_memory_gb=self.hw_profile.total_memory_gb,
                world_size=self.num_gpus,
                vocab_size=tok.vocab_size,
            )
            print(f"[Auto-Tuner] Recommended Micro-Batch Size per GPU: {self.auto_micro_batch}")

    def _resolve_checkpoint(self, stage: str) -> str:
        """Finds checkpoints with priority for nominal_tag, falling back to dynamic size_tag."""
        candidates = [
            f"checkpoints/{stage}_{self.nominal_tag}/best_{'model' if stage == 'pretrain' else stage + '_model'}.pt",
            f"checkpoints/{stage}_{self.nominal_tag}/latest.pt",
            f"checkpoints/{stage}_{self.nominal_tag}/{'final_model' if stage == 'pretrain' else stage + '_final'}.pt",
            f"checkpoints/{stage}_{self.size_tag}/best_{'model' if stage == 'pretrain' else stage + '_model'}.pt",
            f"checkpoints/{stage}_{self.size_tag}/latest.pt",
            f"checkpoints/{stage}_{self.size_tag}/{'final_model' if stage == 'pretrain' else stage + '_final'}.pt",
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]

    @property
    def pretrain_ckpt(self) -> str:
        return self._resolve_checkpoint("pretrain")

    @property
    def sft_ckpt(self) -> str:
        return self._resolve_checkpoint("sft")

    @property
    def dpo_ckpt(self) -> str:
        return self._resolve_checkpoint("dpo")

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
            if "model_state_dict" not in ckpt and "model" not in ckpt and not isinstance(ckpt, dict):
                print(f"❌ Health Gate Error: Invalid checkpoint structure at '{path}'")
                return False
            return True
        except Exception as e:
            print(f"❌ Health Gate Error: Failed to deserialize '{path}': {e}")
            return False

    def ensure_data_prepared(self):
        """Stage 0: Autonomous Cold-Start Data Ingestion & Packing Gate."""
        self.print_stage_banner("0. Autonomous Data Ingestion & Verification")

        # 0A. Pre-training Shard Check & Validation Shard Resilience
        train_shards = sorted(glob.glob(os.path.join(self.cfg.pretrain_data_dir, "train_*.bin")))
        has_train = bool(os.path.exists(os.path.join(self.cfg.pretrain_data_dir, "train.bin")) or train_shards)
        val_shards = sorted(glob.glob(os.path.join(self.cfg.pretrain_data_dir, "val_*.bin")))
        has_val = bool(os.path.exists(os.path.join(self.cfg.pretrain_data_dir, "val.bin")) or val_shards)

        # Auto-link last train shard if val shard is missing
        if has_train and not has_val and train_shards:
            target_val = os.path.join(self.cfg.pretrain_data_dir, "val_00000.bin")
            try:
                os.symlink(os.path.abspath(train_shards[-1]), target_val)
                print(f"✓ Linked validation shard: '{os.path.basename(train_shards[-1])}' -> '{target_val}'")
                has_val = True
            except Exception as e:
                print(f"⚠️ Failed to symlink val shard ({e}).")

        if not (has_train and has_val) and self.cfg.resume_from not in ("sft", "dpo"):
            print(f"⚠️ Pre-training binary shards missing in '{self.cfg.pretrain_data_dir}'.")
            val_budget = max(300_000, min(500_000, self.cfg.bootstrap_pretrain_tokens // 4))
            print(f"   Executing token packing curriculum target: {self.cfg.bootstrap_pretrain_tokens:,} tokens...")
            cmd = [
                sys.executable,
                "-m", "data.prepare_packed_curriculum",
                f"output_dir={self.cfg.pretrain_data_dir}",
                f"total_tokens={self.cfg.bootstrap_pretrain_tokens}",
                f"val_tokens={val_budget}",
            ]
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                raise RuntimeError("Stage 0: Pre-training token packing failed.")
        else:
            print(f"✓ Pre-training shards verified in '{self.cfg.pretrain_data_dir}'")

        # 0B. SFT Dialogue Check
        if not os.path.exists(self.cfg.sft_data_path) and self.cfg.resume_from != "dpo":
            print(f"⚠️ SFT dataset missing at '{self.cfg.sft_data_path}'. Executing composite harvest...")
            out_dir = os.path.dirname(self.cfg.sft_data_path) or "data/sft"
            cmd = [
                sys.executable,
                "-m", "data.prepare_sft",
                f"output_dir={out_dir}",
                f"total_samples={self.cfg.bootstrap_sft_samples}",
            ]
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                raise RuntimeError("Stage 0: SFT dataset preparation failed.")
        else:
            print(f"✓ SFT dataset verified at '{self.cfg.sft_data_path}'")

        # 0C. DPO Preference Check
        dpo_missing = not os.path.exists(self.cfg.dpo_data_path)
        dpo_stub = False
        if not dpo_missing:
            size_bytes = os.path.getsize(self.cfg.dpo_data_path)
            if size_bytes < 50_000:
                dpo_stub = True

        if dpo_missing or dpo_stub:
            reason = "missing" if dpo_missing else "minimal smoke stub (<50KB)"
            print(f"⚠️ DPO dataset at '{self.cfg.dpo_data_path}' is {reason}. Executing composite harvest...")
            out_dir = os.path.dirname(self.cfg.dpo_data_path) or "data/dpo"
            os.makedirs(out_dir, exist_ok=True)

            cmd = [
                sys.executable,
                "-m", "data.prepare_dpo",
                f"output_path={self.cfg.dpo_data_path}",
                f"total_samples={self.cfg.bootstrap_dpo_samples}",
            ]
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                raise RuntimeError("Stage 0: DPO composite dataset preparation failed.")
        else:
            print(f"✓ DPO composite dataset verified at '{self.cfg.dpo_data_path}'")

    def run_pretrain(self):
        pretrain_title = (
            f"1. Pre-training (Multi-GPU DDP - {self.num_gpus} GPUs)"
            if self.num_gpus > 1
            else "1. Pre-training (Single-GPU torchrun)"
        )
        self.print_stage_banner(pretrain_title)
        if self.cfg.resume_from in ("sft", "dpo"):
            print(f"⏩ Skipping Pre-training (resuming from stage '{self.cfg.resume_from}').")
            return

        # Pre-flight fail-fast token budget validation before spawning torchrun
        target_tokens = parse_token_string(self.cfg.tokens)
        effective_max_steps = self.cfg.max_steps or self.cfg.pretrain_max_steps

        shards = sorted(glob.glob(os.path.join(self.cfg.pretrain_data_dir, "train_*.bin")))
        if not shards and os.path.exists(os.path.join(self.cfg.pretrain_data_dir, "train.bin")):
            shards = [os.path.join(self.cfg.pretrain_data_dir, "train.bin")]

        if shards:
            available_tokens = sum(os.path.getsize(s) // 2 for s in shards)
            if target_tokens is not None and target_tokens > available_tokens:
                print("\n" + "=" * 70, file=sys.stderr)
                print(f"[FATAL CONFIG ERROR] Insufficient tokens in '{self.cfg.pretrain_data_dir}'!", file=sys.stderr)
                print(f"  Requested Token Budget : {target_tokens:,} tokens ({target_tokens / 1e9:.2f}B)", file=sys.stderr)
                print(f"  Available on Disk      : {available_tokens:,} tokens ({available_tokens / 1e9:.2f}B) across {len(shards)} shards", file=sys.stderr)
                print(f"  Deficit                : {target_tokens - available_tokens:,} tokens", file=sys.stderr)
                print("=" * 70 + "\n", file=sys.stderr)
                raise RuntimeError(
                    f"Pre-training aborted: requested {target_tokens/1e9:.2f}B tokens, "
                    f"but only {available_tokens/1e9:.2f}B exist on disk."
                )

        output_dir = self.cfg.output_dir or f"checkpoints/pretrain_{self.nominal_tag}"

        cmd = [
            "torchrun",
            f"--nproc_per_node={self.num_gpus}",
            "-m", "pretrain.train",
            f"size={self.nominal_tag}",
            f"data_dir={self.cfg.pretrain_data_dir}",
            f"output_dir={output_dir}",
            f"tokenizer_type={self.cfg.tokenizer_type}",
            f"micro_batch_size={self.auto_micro_batch}",
        ]
        if self.cfg.global_batch_size:
            cmd.append(f"global_batch_size={self.cfg.global_batch_size}")
        if self.cfg.tokens:
            cmd.append(f"tokens={self.cfg.tokens}")
        if effective_max_steps:
            cmd.append(f"max_steps={effective_max_steps}")
        if self.cfg.tokenizer_path:
            cmd.append(f"tokenizer_path={self.cfg.tokenizer_path}")

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

        sft_batch = min(16, self.auto_micro_batch)
        cmd = [
            sys.executable,
            "-m", "sft.train",
            f"pretrained_ckpt={self.pretrain_ckpt}",
            f"output_dir=checkpoints/sft_{self.nominal_tag}",
            f"data_path={self.cfg.sft_data_path}",
            f"epochs={self.cfg.sft_epochs}",
            f"tokenizer_type={self.cfg.tokenizer_type}",
            f"per_device_batch_size={sft_batch}",
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

        dpo_batch = max(1, min(8, self.auto_micro_batch // 2))

        cmd = [
            sys.executable,
            "-m", "dpo.train",
            f"sft_model_path={self.sft_ckpt}",
            f"output_dir=checkpoints/dpo_{self.nominal_tag}",
            f"data_path={self.cfg.dpo_data_path}",
            f"epochs={self.cfg.dpo_epochs}",
            f"learning_rate={self.cfg.dpo_learning_rate}",
            f"tokenizer_type={self.cfg.tokenizer_type}",
            f"per_device_batch_size={dpo_batch}",
        ]
        if self.cfg.tokenizer_path:
            cmd.append(f"tokenizer_path={self.cfg.tokenizer_path}")
        if self.cfg.dpo_max_steps:
            cmd.append(f"max_steps={self.cfg.dpo_max_steps}")

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
        print(f"Model Size Target:      {self.nominal_tag} (Tag: {self.size_tag})")
        print(f"Hardware Profile:       {self.hw_profile.device_name} ({self.hw_profile.arch_generation})")
        print(f"Pre-training Compute:   {'Multi-GPU' if self.num_gpus > 1 else 'Single-GPU'} ({self.num_gpus} GPU{'s' if self.num_gpus > 1 else ''} via torchrun)")
        print(f"Alignment Compute:      Single-GPU (SFT & DPO)")
        print("=" * 70)

        target_stage = (self.cfg.stage or "").lower()

        if target_stage in ("prepare", "stage0", "data"):
            self.ensure_data_prepared()
            return

        # Stage 0: Data verification & bootstrap
        self.ensure_data_prepared()

        # Handle stage selection or full sequence
        if target_stage == "pretrain":
            self.run_pretrain()
        elif target_stage == "sft":
            self.run_sft()
        elif target_stage == "dpo":
            self.run_dpo()
        else:
            self.run_pretrain()
            self.run_sft()
            self.run_dpo()

        elapsed_mins = (time.time() - start_time) / 60.0
        print("\n" + "=" * 70)
        print(f"✓ Pipeline Execution Completed in {elapsed_mins:.2f} minutes.")
        final_target = self.pretrain_ckpt if target_stage == "pretrain" else (self.sft_ckpt if target_stage == "sft" else self.dpo_ckpt)
        print(f"Final Artifact: {final_target}")
        print("=" * 70)


def main():
    cfg = parse_cli_args(PipelineConfig)
    orchestrator = PipelineOrchestrator(cfg)
    orchestrator.execute()


if __name__ == "__main__":
    main()