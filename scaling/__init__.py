"""scaling: Dynamic architecture scaling, hardware profiling, and runtime configuration."""

from .config import RuntimeConfig
from .engine import (
    count_parameters,
    derive_learning_rate,
    derive_model_config,
    format_size,
    parse_size_str,
)

__all__ = [
    "RuntimeConfig",
    "count_parameters",
    "derive_learning_rate",
    "derive_model_config",
    "format_size",
    "parse_size_str",
]