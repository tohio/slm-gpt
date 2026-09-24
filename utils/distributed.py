"""
utils/distributed.py: Distributed Data Parallel (DDP) utilities for slm-gpt pre-training.
Handles process group lifecycle, rank-0 I/O safety, tensor reduction, and single-GPU fallback.
"""

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist


def init_distributed() -> Tuple[int, int, int, bool, torch.device]:
    """
    Initializes DDP if launched under torchrun, or falls back to single-device mode.

    Returns:
        rank: Global process rank (0 to world_size - 1)
        local_rank: GPU index on the current node
        world_size: Total number of processes across all nodes
        is_distributed: True if running multi-GPU DDP
        device: Active torch.device for this rank
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )
        is_distributed = True
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        is_distributed = False
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return rank, local_rank, world_size, is_distributed, device


def is_main_process(rank: Optional[int] = None) -> bool:
    """Returns True if the current process is Rank 0."""
    if rank is not None:
        return rank == 0
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def barrier():
    """Synchronizes all processes across the cluster."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def print_rank0(*args, **kwargs):
    """Prints only from global Rank 0 to avoid console spam."""
    if is_main_process():
        print(*args, **kwargs)


def cleanup_distributed():
    """Destroys the process group on clean shutdown."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """All-reduces a tensor across all ranks for global metric tracking."""
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    rt = tensor.detach().clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if average:
        rt /= dist.get_world_size()
    return rt


def save_checkpoint_rank0(state_dict: Dict[str, Any], path: str):
    """Saves model checkpoints strictly from Rank 0, with a barrier."""
    if is_main_process():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)
    barrier()