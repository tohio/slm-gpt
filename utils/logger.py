"""
utils/logger.py: Rank-0 Safe Weights & Biases Logger.
Loads WANDB_API_KEY, WANDB_PROJECT, and WANDB_RUN_NAME directly from .env via load_env_file.
"""

from dataclasses import asdict, is_dataclass
import os
from pathlib import Path
from typing import Any, Dict, Optional
import torch

from utils.distributed import is_main_process


def load_env_file(env_path: Optional[str] = None) -> None:
    """Reads .env file and populates os.environ without requiring external packages."""
    target = Path(env_path) if env_path else Path.cwd() / ".env"
    if not target.exists():
        # Check parent folders
        for parent in target.resolve().parents:
            candidate = parent / ".env"
            if candidate.exists():
                target = candidate
                break

    if target.exists():
        try:
            with open(target, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v
        except Exception as e:
            print(f"[Logger] Warning reading .env at {target}: {e}")


def _clean_for_json(obj: Any) -> Any:
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, torch.dtype):
        return str(obj).replace("torch.", "")
    if is_dataclass(obj):
        return {k: _clean_for_json(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: _clean_for_json(v) for k, v in vars(obj).items()}
    return str(obj)


class WandbLogger:
    """Centralized W&B Logger reading credentials and settings strictly from .env on Rank 0."""

    def __init__(
        self,
        runtime_cfg: Optional[Any] = None,
        default_project: str = "slm-gpt",
        run_name: Optional[str] = None,
    ):
        load_env_file()
        self.is_main = is_main_process()
        self._wandb = None

        # Pure .env resolution
        api_key = os.environ.get("WANDB_API_KEY")
        explicit_enable = os.environ.get("WANDB_ENABLE", "true").lower() in ("true", "1", "yes")
        self.enabled = self.is_main and explicit_enable and bool(api_key)

        if not self.enabled:
            return

        try:
            import wandb
            self._wandb = wandb

            project = os.environ.get("WANDB_PROJECT", default_project)
            entity = os.environ.get("WANDB_ENTITY", None)
            resolved_run_name = run_name or os.environ.get("WANDB_RUN_NAME", None)
            config_dict = _clean_for_json(runtime_cfg) if runtime_cfg else {}

            self._wandb.init(
                project=project,
                entity=entity,
                name=resolved_run_name,
                config=config_dict,
                dir=os.environ.get("WANDB_DIR", "./"),
            )
            print(f"[Logger] ✓ W&B Connected: {self._wandb.run.name} ({self._wandb.run.url})")
        except ImportError:
            print("[Logger] Notice: 'wandb' package not installed. Skipping W&B logging.")
            self.enabled = False
        except Exception as e:
            print(f"[Logger] Notice: W&B initialization skipped ({e}).")
            self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled or self._wandb is None:
            return
        clean = {
            k: (v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v)
            for k, v in metrics.items()
        }
        self._wandb.log(clean, step=step)

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()