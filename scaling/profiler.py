"""
scaling/profiler.py: Architecture-Aware Environment Bootstrapper & Profiler.
Supports:
  1. NVIDIA Blackwell (B100/B200/GB200) -> FA4 -> FA2 -> Native SDPA
  2. NVIDIA Hopper (H100/H200)          -> FA4 -> FA2 -> Native SDPA
  3. NVIDIA Ampere & Ada (A100/RTX)     -> FA2 -> Native SDPA
  4. Apple Silicon (MPS)                -> Native SDPA
  5. CPU (x86_64, ARM64)                -> Native SDPA
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
    "psutil",
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
        fa_candidates: List[Tuple[str, List[str]]],
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
        if torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(dev_idx)
            props = torch.cuda.get_device_properties(dev_idx)
            total_mem = props.total_memory / (1024 ** 3)
            sm_major, sm_minor = props.major, props.minor
            cap = (sm_major, sm_minor)

            cuda_version = torch.version.cuda or ""
            major_cuda = int(cuda_version.split(".")[0]) if "." in cuda_version else 12

            # --- A. Hopper (SM 9.0) & Blackwell (SM 10.x+) ---
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
                        ("flash_attn.cute", ["install", "--pre", "flash-attn-4", "--no-build-isolation"]),
                        ("flash_attn", ["install", "flash-attn>=2.6.0", "--no-build-isolation"]),
                    ],
                )

            # --- B. Ampere & Ada (A100, RTX 3090/4090 - SM 8.0 to 8.9) ---
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
                        ("flash_attn", ["install", "flash-attn>=2.6.0", "--no-build-isolation"]),
                    ],
                )

            # --- C. Older CUDA Architectures (< SM 8.0) ---
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

        # --- D. Apple Silicon (MPS) ---
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return HardwareProfile(
                device_type="mps",
                device_name=f"Apple Silicon ({platform.processor() or 'ARM'})",
                arch_generation="Apple Metal (MPS)",
                total_memory_gb=16.0,
                compute_capability=None,
                precision=torch.float32,
                precision_str="float32",
                wheel_index_url=None,
                recommended_packages=BASE_DEPENDENCIES + ["torch>=2.4.0"],
                fa_candidates=[],
            )

        # --- E. CPU Fallback ---
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
            fa_candidates=[],
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
        profile = cls.profile()

        # Step 1: Verify core dependencies
        core_checks = {
            "regex": "regex",
            "numpy": "numpy",
            "psutil": "psutil",
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

        # Step 3: Check if an FA candidate is already installed and importable
        for mod_name, _ in profile.fa_candidates:
            root_mod = mod_name.split(".")[0]
            if importlib.util.find_spec(root_mod) is not None:
                try:
                    __import__(mod_name)
                    return
                except Exception:
                    pass

        # Step 4: Probing candidate backends in priority order
        for mod_name, pip_args in profile.fa_candidates:
            pkg_desc = " ".join(pip_args[1:])
            print(f"[HardwareProfiler] Probing hardware backend candidate: {pkg_desc}...")
            try:
                cmd = [sys.executable, "-m", "pip"] + pip_args
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if res.returncode == 0:
                    print(f"[HardwareProfiler] ✓ Successfully activated {mod_name}.")
                    return
                else:
                    print(f"[HardwareProfiler] Candidate {pkg_desc} failed. Trying next...")
            except Exception as e:
                print(f"[HardwareProfiler] Build error for {pkg_desc}: {e}")

        print("[HardwareProfiler] Notice: Operating under PyTorch-Native-SDPA.")