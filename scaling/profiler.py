"""
scaling/profiler.py: Architecture-Aware Environment Bootstrapper & Profiler.
Supports:
  1. NVIDIA Blackwell (B100/B200/GB200, SM 10.x/12.x) -> FA4 / FA3 / FA2
  2. NVIDIA Hopper (H100/H200, SM 9.0)                -> FA4 / FA3 / FA2
  3. NVIDIA Ampere & Ada (A100/RTX 3090/4090, SM 8.x) -> FA2
  4. Apple Silicon (M1/M2/M3/M4 via MPS)              -> Native PyTorch SDPA
  5. CPU (x86_64, ARM64)                              -> Native PyTorch SDPA
"""

import importlib.util
import os
import platform
import subprocess
import sys
from typing import List, Optional, Tuple

import torch

BASE_DEPENDENCIES: List[str] = [
    "regex",
    "datasets>=2.14.0",
    "python-dotenv>=1.0.0",
    "packaging>=23.0",
    "ninja>=1.11.0",
    "wandb",
    "numpy>=1.26.0",
    "tiktoken>=0.7.0",
    "safetensors>=0.4.4",
    "huggingface_hub>=0.24.0",
    "transformers>=4.44.0",
    "pytest>=8.0.0",
]


class HardwareProfile:
    def __init__(
        self,
        device_type: str,
        device_name: str,
        arch_generation: str,
        total_memory_gb: float,
        compute_capability: Optional[Tuple[int, int]],
        precision: torch.dtype,
        precision_str: str,
        wheel_index_url: Optional[str],
        recommended_packages: List[str],
        fa_candidates: List[Tuple[str, str]],
    ):
        self.device_type = device_type
        self.device_name = device_name
        self.arch_generation = arch_generation
        self.total_memory_gb = total_memory_gb
        self.compute_capability = compute_capability
        self.precision = precision
        self.precision_str = precision_str
        self.wheel_index_url = wheel_index_url
        self.recommended_packages = recommended_packages
        self.fa_candidates = fa_candidates


class HardwareProfiler:
    @staticmethod
    def profile() -> HardwareProfile:
        # =================================================================
        # 1. NVIDIA CUDA PLATFORMS (A100 and up: SM 8.0 -> SM 12.0)
        # =================================================================
        if torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(dev_idx)
            props = torch.cuda.get_device_properties(dev_idx)
            total_mem = props.total_memory / (1024 ** 3)
            sm_major, sm_minor = props.major, props.minor
            cap = (sm_major, sm_minor)

            cuda_version = torch.version.cuda or ""
            major_cuda = int(cuda_version.split(".")[0]) if "." in cuda_version else 12

            # Tier 1: NVIDIA Blackwell (SM 10.x / 12.x) & Hopper (SM 9.0)
            if sm_major >= 9:
                arch = f"NVIDIA Blackwell ({dev_name})" if sm_major >= 10 else f"NVIDIA Hopper ({dev_name})"
                wheel_url = "https://download.pytorch.org/whl/cu130" if major_cuda >= 13 else "https://download.pytorch.org/whl/cu126"

                return HardwareProfile(
                    device_type="cuda",
                    device_name=dev_name,
                    arch_generation=arch,
                    total_memory_gb=total_mem,
                    compute_capability=cap,
                    precision=torch.bfloat16,
                    precision_str="bfloat16",
                    wheel_index_url=wheel_url,
                    recommended_packages=BASE_DEPENDENCIES + ["torch>=2.5.0", "triton>=3.1.0"],
                    fa_candidates=[
                        ("flash_attn_4", "flash-attn-4"),
                        ("flash_attn_interface", "flash-attn-3"),
                        ("flash_attn", "flash-attn>=2.6.0"),
                    ],
                )

            # Tier 2: NVIDIA Ampere (A100) & Ada Lovelace (SM 8.0 - 8.9)
            elif sm_major == 8:
                arch = f"NVIDIA Ampere/Ada ({dev_name})"
                wheel_url = "https://download.pytorch.org/whl/cu124"

                return HardwareProfile(
                    device_type="cuda",
                    device_name=dev_name,
                    arch_generation=arch,
                    total_memory_gb=total_mem,
                    compute_capability=cap,
                    precision=torch.bfloat16,
                    precision_str="bfloat16",
                    wheel_index_url=wheel_url,
                    recommended_packages=BASE_DEPENDENCIES + ["torch>=2.4.0", "triton>=3.0.0"],
                    fa_candidates=[
                        ("flash_attn", "flash-attn>=2.6.0"),
                    ],
                )

            # Fallback for older CUDA devices (< SM 8.0)
            else:
                return HardwareProfile(
                    device_type="cuda",
                    device_name=dev_name,
                    arch_generation="NVIDIA Legacy CUDA",
                    total_memory_gb=total_mem,
                    compute_capability=cap,
                    precision=torch.float16,
                    precision_str="float16",
                    wheel_index_url="https://download.pytorch.org/whl/cu121",
                    recommended_packages=BASE_DEPENDENCIES + ["torch>=2.4.0"],
                    fa_candidates=[],
                )

        # =================================================================
        # 2. APPLE SILICON PLATFORMS (M1, M2, M3, M4 via MPS)
        # =================================================================
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return HardwareProfile(
                device_type="mps",
                device_name=f"Apple Silicon ({platform.processor() or 'ARM'})",
                arch_generation="Apple Metal (MPS)",
                total_memory_gb=16.0,  # Unified memory baseline
                compute_capability=None,
                precision=torch.float32,
                precision_str="float32",
                wheel_index_url=None,
                recommended_packages=BASE_DEPENDENCIES + ["torch>=2.4.0"],
                fa_candidates=[],  # Handled natively by Apple Metal SDPA
            )

        # =================================================================
        # 3. CPU PLATFORMS (x86_64, ARM64)
        # =================================================================
        return HardwareProfile(
            device_type="cpu",
            device_name=platform.processor() or "Generic CPU",
            arch_generation="CPU Execution",
            total_memory_gb=8.0,
            compute_capability=None,
            precision=torch.float32,
            precision_str="float32",
            wheel_index_url="https://download.pytorch.org/whl/cpu",
            recommended_packages=BASE_DEPENDENCIES + ["torch>=2.4.0"],
            fa_candidates=[],  # Handled natively by PyTorch C++ Vectorized SDPA
        )

    @classmethod
    def generate_requirements_file(cls, output_path: str = "requirements.txt") -> str:
        profile = cls.profile()
        lines = [
            "# ===================================================================",
            "# slm-gpt Architecture-Generated Dependencies",
            f"# Hardware:          {profile.device_name}",
            f"# Architecture:      {profile.arch_generation}",
            f"# Total Memory:      {profile.total_memory_gb:.1f} GB",
            f"# Precision Target:  {profile.precision_str}",
            "# ===================================================================\n",
        ]

        if profile.wheel_index_url:
            lines.append(f"--extra-index-url {profile.wheel_index_url}\n")

        for pkg in profile.recommended_packages:
            lines.append(pkg)

        content = "\n".join(lines) + "\n"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[HardwareProfiler] Wrote dependencies for {profile.arch_generation} to {output_path}")
        return output_path

    @classmethod
    def auto_install_environment(cls):
        """
        Autonomous Multi-Platform Environment Bootstrapper:
        1. Checks core dependencies (regex, tiktoken, datasets, wandb, etc.).
        2. Auto-installs missing packages without human intervention.
        3. Probes hardware-specific FlashAttention candidates (A100+: FA2, Hopper/Blackwell: FA4->FA3->FA2).
        4. Reverts cleanly to PyTorch Native SDPA on MPS, CPU, or compiler misses.
        """
        profile = cls.profile()

        # Step 1: Ensure core runtime packages are present
        core_checks = {
            "regex": "regex",
            "numpy": "numpy",
            "tiktoken": "tiktoken",
            "datasets": "datasets",
            "wandb": "wandb",
            "transformers": "transformers",
        }
        missing_core = [pkg for pkg, mod in core_checks.items() if importlib.util.find_spec(mod) is None]

        if missing_core:
            print(f"[HardwareProfiler] Cold-start detected on {profile.arch_generation}.")
            print(f"[HardwareProfiler] Missing core packages: {missing_core}. Auto-installing...")
            req_file = cls.generate_requirements_file()
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
            print("[HardwareProfiler] ✓ Core packages verified.")

        # Step 2: Skip FA checks for MPS and CPU
        if not profile.fa_candidates:
            return

        # Step 3: Check if an FA backend is already active
        for mod_name, _ in profile.fa_candidates:
            if importlib.util.find_spec(mod_name) is not None:
                return

        # Step 4: Multi-tier installation probe for A100 and up
        for mod_name, pip_pkg in profile.fa_candidates:
            print(f"[HardwareProfiler] Probing hardware backend candidate: {pip_pkg}...")
            try:
                res = subprocess.run(
                    [sys.executable, "-m", "pip", "install", pip_pkg, "--no-build-isolation"],
                    capture_output=True,
                    text=True,
                    timeout=240,
                )
                if res.returncode == 0:
                    print(f"[HardwareProfiler] ✓ Successfully activated {pip_pkg}.")
                    return
            except Exception as e:
                pass

        print("[HardwareProfiler] Notice: Operating under PyTorch-Native-SDPA.")


def tune_micro_batch_size(
    target_params: int,
    max_seq_len: int,
    total_memory_gb: float,
    world_size: int = 1,
    vocab_size: int = 50304,
) -> int:
    """Auto-tunes micro-batch size based on device memory and sequence length."""
    static_gb = (target_params * 14) / (1024 ** 3) + 2.0
    usable_vram = max(0.5, (total_memory_gb * 0.75) - static_gb)

    vocab_workspace_bytes = max_seq_len * vocab_size * 10
    activation_bytes = max_seq_len * 768 * 90
    per_sample_gb = (vocab_workspace_bytes + activation_bytes) / (1024 ** 3)

    estimated_batch = int(usable_vram / max(per_sample_gb, 0.1))

    powers = [16, 8, 4, 2, 1] if max_seq_len >= 2048 else [32, 16, 8, 4, 2, 1]
    for p in powers:
        if estimated_batch >= p:
            return p
    return 1


if __name__ == "__main__":
    HardwareProfiler.auto_install_environment()