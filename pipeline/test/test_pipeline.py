"""
pipeline/test/test_pipeline.py: End-to-end integration and health-gate tests.
"""

from pathlib import Path
import pytest

from pipeline.orchestrator import PipelineConfig, PipelineOrchestrator


@pytest.mark.parametrize("size", ["125M", "350M"])
def test_health_gate_verification(size: str):
    """Verify health gate checks fail cleanly on missing checkpoints regardless of model size."""
    cfg = PipelineConfig(
        size=size,
        tokenizer_type="tiktoken",
        pretrain_data_dir="data/pretrain",
        sft_data_path="data/sft/train.jsonl",
        dpo_data_path="data/dpo/preference_pairs.jsonl",
    )
    orch = PipelineOrchestrator(cfg)

    fake_ckpt = Path("checkpoints/non_existent_stage/model.pt")
    assert orch.verify_health_gate(fake_ckpt) is False