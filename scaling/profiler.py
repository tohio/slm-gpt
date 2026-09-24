"""
scaling/profiler.py: Hardware Profiler & Architecture-Aware Requirements Generator.
Detects CUDA 13.0+ environments, Blackwell (B200/B300), Hopper (H100/H200),
Ampere, Apple Silicon (MPS), and CPU platforms.
Dynamically sets precision, batching, and CUDA 13.0 wheel repositories.
"""

import os
import platform
import sys
from typing import List, Optional, Tuple

import torch

BASE_PACKAGES: List[str] = [
    "regex",
    "datasets>=2.14.0",
    "python-dotenv>=1.0.0",
    "packaging>=23.0",
    "ninja>=1.11.0",
    "wandb",
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


class HardwareProfiler:
    @staticmethod
    def profile() -> HardwareProfile:
        # 1. CUDA Hardware Detection
        if torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(dev_idx)
            props = torch.cuda.get_device_properties(dev_idx)
            total_mem = props.total_memory / (1024 ** 3)
            sm_major, sm_minor = props.major, props.minor
            cap = (sm_major, sm_minor)

            # Inspect runtime driver version for explicit CUDA 13 support
            cuda_version = torch.version.cuda or ""
            major_cuda = int(cuda_version.split(".")[0]) if "." in cuda_version else 12

            # --- A. Blackwell & Blackwell Ultra (SM 10.x, SM 12.x) / CUDA 13.0+ ---
            if sm_major >= 10 or major_cuda >= 13:
                if sm_major == 10 and sm_minor >= 3:
                    arch = "NVIDIA Blackwell Ultra (B300/GB300)"
                elif sm_major == 10:
                    arch = "NVIDIA Blackwell (B200/GB200)"
                else:
                    arch = f"NVIDIA Architecture (SM {sm_major}.{sm_minor} / CUDA {major_cuda})"

                return HardwareProfile(
                    device_type="cuda",
                    device_name=dev_name,
                    arch_generation=arch,
                    total_memory_gb=total_mem,
                    compute_capability=cap,
                    precision=torch.bfloat16,
                    precision_str="bfloat16",
                    wheel_index_url="https://download.pytorch.org/whl/cu130",
                    recommended_packages=BASE_PACKAGES + [
                        "torch>=2.5.0",
                        "numpy>=1.26.0",
                        "tiktoken>=0.7.0",
                        "safetensors>=0.4.4",
                        "huggingface_hub>=0.24.0",
                        "transformers>=4.44.0",
                        "triton>=3.1.0",
                    ],
                )

            # --- B. Hopper (SM 9.0: H100/H200) & Ada Lovelace (SM 8.9) ---
            elif sm_major == 9 or (sm_major == 8 and sm_minor >= 9):
                arch = "NVIDIA Hopper (H100/H200)" if sm_major == 9 else "NVIDIA Ada Lovelace"
                return HardwareProfile(
                    device_type="cuda",
                    device_name=dev_name,
                    arch_generation=arch,
                    total_memory_gb=total_mem,
                    compute_capability=cap,
                    precision=torch.bfloat16,
                    precision_str="bfloat16",
                    wheel_index_url="https://download.pytorch.org/whl/cu126",
                    recommended_packages=BASE_PACKAGES + [
                        "torch>=2.4.0",
                        "numpy>=1.26.0",
                        "tiktoken>=0.7.0",
                        "safetensors>=0.4.4",
                        "huggingface_hub>=0.24.0",
                        "transformers>=4.44.0",
                        "triton>=3.0.0",
                    ],
                )

            # --- C. Ampere (SM 8.0/8.6: A100 / RTX 3090) ---
            elif sm_major == 8:
                return HardwareProfile(
                    device_type="cuda",
                    device_name=dev_name,
                    arch_generation="NVIDIA Ampere (A100/RTX 3090)",
                    total_memory_gb=total_mem,
                    compute_capability=cap,
                    precision=torch.bfloat16,
                    precision_str="bfloat16",
                    wheel_index_url="https://download.pytorch.org/whl/cu124",
                    recommended_packages=BASE_PACKAGES + [
                        "torch>=2.4.0",
                        "numpy>=1.26.0",
                        "tiktoken>=0.7.0",
                        "safetensors>=0.4.4",
                        "huggingface_hub>=0.24.0",
                        "transformers>=4.44.0",
                        "triton>=3.0.0",
                    ],
                )

            # --- D. Turing & Volta (SM 7.0/7.5) ---
            else:
                return HardwareProfile(
                    device_type="cuda",
                    device_name=dev_name,
                    arch_generation="NVIDIA Legacy (Volta/Turing)",
                    total_memory_gb=total_mem,
                    compute_capability=cap,
                    precision=torch.float16,
                    precision_str="float16",
                    wheel_index_url="https://download.pytorch.org/whl/cu121",
                    recommended_packages=BASE_PACKAGES + [
                        "torch>=2.4.0",
                        "numpy>=1.26.0",
                        "tiktoken>=0.7.0",
                        "safetensors>=0.4.4",
                        "huggingface_hub>=0.24.0",
                        "transformers>=4.44.0",
                    ],
                )

        # 2. Apple Silicon (MPS Backend)
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
                recommended_packages=BASE_PACKAGES + [
                    "torch>=2.4.0",
                    "numpy>=1.26.0",
                    "tiktoken>=0.7.0",
                    "safetensors>=0.4.4",
                    "huggingface_hub>=0.24.0",
                    "transformers>=4.44.0",
                ],
            )

        # 3. CPU Fallback
        return HardwareProfile(
            device_type="cpu",
            device_name=platform.processor() or "Generic CPU",
            arch_generation="CPU Architecture",
            total_memory_gb=8.0,
            compute_capability=None,
            precision=torch.float32,
            precision_str="float32",
            wheel_index_url="https://download.pytorch.org/whl/cpu",
            recommended_packages=BASE_PACKAGES + [
                "torch>=2.4.0",
                "numpy>=1.26.0",
                "tiktoken>=0.7.0",
                "safetensors>=0.4.4",
                "huggingface_hub>=0.24.0",
                "transformers>=4.44.0",
            ],
        )

    @classmethod
    def generate_requirements_file(cls, output_path: str = "requirements.txt") -> str:
        """Inspects the current system and generates an architecture-tailored requirements.txt."""
        profile = cls.profile()
        lines = [
            "# ===================================================================",
            "# slm-gpt Architecture-Generated Dependencies",
            f"# Hardware:          {profile.device_name}",
            f"# Architecture:      {profile.arch_generation}",
            f"# Total Memory:      {profile.total_memory_gb:.1f} GB",
            f"# Precision Target:  {profile.precision_str}",
        ]
        if profile.compute_capability:
            lines.append(f"# CUDA SM:           {profile.compute_capability[0]}.{profile.compute_capability[1]}")
        lines.append("# ===================================================================\n")

        if profile.wheel_index_url:
            lines.append(f"--extra-index-url {profile.wheel_index_url}\n")

        for pkg in profile.recommended_packages:
            lines.append(pkg)

        lines.append("pytest>=8.0.0")

        content = "\n".join(lines) + "\n"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[HardwareProfiler] Wrote dependencies for {profile.arch_generation} to {output_path}")
        return output_path


def tune_micro_batch_size(
    target_params: int,
    max_seq_len: int,
    total_memory_gb: float,
    world_size: int = 1,
) -> int:
    """Auto-tunes micro-batch sizes based on VRAM capacity."""
    bytes_per_token = 2  # bfloat16
    model_bytes = target_params * 2
    optim_bytes = target_params * 8  # AdamW fp32 states
    static_bytes = (model_bytes + optim_bytes) / (1024 ** 3)

    available_vram = max(1.0, total_memory_gb - static_bytes)
    activation_bytes_per_sample = (max_seq_len * bytes_per_token * 96) / (1024 ** 3)

    estimated_batch = int(available_vram / max(activation_bytes_per_sample, 0.01))

    powers = [128, 64, 32, 16, 8, 4, 2, 1]
    for p in powers:
        if estimated_batch >= p:
            return p
    return 1


if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "requirements.txt"
    HardwareProfiler.generate_requirements_file(target_file)