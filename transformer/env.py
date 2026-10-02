import os
from dotenv import load_dotenv

load_dotenv()

def get_model_tag(default: str = "125M") -> str:
    """Returns MODEL_TAG from environment, defaulting to 125M."""
    return os.getenv("MODEL_TAG", default).strip()

def is_training_inference_enabled(default: bool = True) -> bool:
    """
    Returns True if TRAINING_INFERENCE is set to true, 1, yes, or t.
    Returns False if set to false, 0, no, or f.
    Defaults to True if unset.
    """
    val = os.getenv("TRAINING_INFERENCE")
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "y", "t")
