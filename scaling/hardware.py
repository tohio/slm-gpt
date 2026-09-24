"""
scaling/hardware.py: Hardware environment profiler, precision selection, and VRAM auto-tuning.
"""

from dataclasses import dataclass
import platform
from typing import Tuple
import torch


@dataclass
class HardwareProfile:
    device_name: str
    vram_gb: float
    compute_capability: Tuple[int, int]
    precision: torch.dtype
    precision_str: str
    recommended_micro_batch: int


def profile_hardware(max_seq_len: int = 2048, d_model: int = 768) -> HardwareProfile:
    """
    Inspects host hardware and determines optimal execution parameters.
    """
    # 1. CUDA Backend
    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_idx)
        vram_gb = round(props.total_memory / (1024**3), 2)
        major, minor = props.major, props.minor

        # Precision auto-selection:
        # Compute Capability >= 8.0 (Ampere, Ada, Hopper, Blackwell) natively supports bfloat16
        if major >= 8:
            precision = torch.bfloat16
            precision_str = "bfloat16"
        elif major >= 7:
            precision = torch.float16
            precision_str = "float16"
        else:
            precision = torch.float32
            precision_str = "float32"

        # Micro-batch auto-tuning based on VRAM footprint
        if vram_gb >= 70.0:
            rec_batch = 16
        elif vram_gb >= 35.0:
            rec_batch = 8
        elif vram_gb >= 15.0:
            rec_batch = 4
        elif vram_gb >= 7.0:
            rec_batch = 2
        else:
            rec_batch = 1

        # Adjust for larger sequence lengths or widths
        if max_seq_len > 2048 or d_model > 1024:
            rec_batch = max(1, rec_batch // 2)

        return HardwareProfile(
            device_name=props.name,
            vram_gb=vram_gb,
            compute_capability=(major, minor),
            precision=precision,
            precision_str=precision_str,
            recommended_micro_batch=rec_batch,
        )

    # 2. Apple Silicon MPS Backend
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return HardwareProfile(
            device_name=f"Apple Silicon ({platform.processor() or 'ARM'})",
            vram_gb=16.0,
            compute_capability=(0, 0),
            precision=torch.float32,
            precision_str="float32",
            recommended_micro_batch=4,
        )

    # 3. CPU Fallback
    return HardwareProfile(
        device_name=platform.processor() or "Generic CPU",
        vram_gb=0.0,
        compute_capability=(0, 0),
        precision=torch.float32,
        precision_str="float32",
        recommended_micro_batch=1,
    )


if __name__ == "__main__":
    profile = profile_hardware()
    print("=" * 60)
    print("          slm-gpt Hardware Profile Detection          ")
    print("=" * 60)
    print(f"Device Name:            {profile.device_name}")
    print(f"Compute Capability:     {profile.compute_capability[0]}.{profile.compute_capability[1]}")
    print(f"Total VRAM:             {profile.vram_gb} GB")
    print(f"Selected Precision:     {profile.precision_str}")
    print(f"Recommended Micro-batch:{profile.recommended_micro_batch}")
    print("=" * 60)