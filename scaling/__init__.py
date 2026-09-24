"""scaling: Dynamic architecture scaling, hardware profiling, and runtime configuration."""

from .config import RuntimeConfig
from .engine import (
    count_parameters,
    derive_learning_rate,
    derive_model_config,
    format_size,
    parse_size_str,
)
from .hardware import HardwareProfile, profile_hardware

__all__ = [
    "RuntimeConfig",
    "HardwareProfile",
    "count_parameters",
    "derive_learning_rate",
    "derive_model_config",
    "format_size",
    "parse_size_str",
    "profile_hardware",
]