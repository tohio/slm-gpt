"""
pipeline/test/test_pipeline.py: Unit tests for PipelineOrchestrator.
Validates path resolution, checkpoint health gate verification, and stage isolation.
"""

from pathlib import Path
import sys
import tempfile
import torch

# Ensure repository root is on sys.path before local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.run import PipelineConfig, PipelineOrchestrator


def test_orchestrator_path_resolution():
    """Verify that dynamic checkpoint and data paths resolve cleanly."""
    cfg = PipelineConfig(
        size="125M",
        tokenizer_type="tiktoken",
        pretrain_data_dir="data/pretrain",
        sft_data_path="data/sft/train.jsonl",
        dpo_data_path="data/dpo/preference_pairs.jsonl",
    )
    orch = PipelineOrchestrator(cfg)

    assert orch.size_tag == "125M"
    assert orch.pretrain_ckpt == "checkpoints/pretrain_125M/best_model.pt"
    assert orch.sft_ckpt == "checkpoints/sft_125M/sft_final.pt"
    assert orch.dpo_ckpt == "checkpoints/dpo_125M/dpo_final.pt"


def test_health_gate_verification():
    """Verify that checkpoint health verification detects valid and invalid files."""
    cfg = PipelineConfig(size="125M")
    orch = PipelineOrchestrator(cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Non-existent file
        assert orch.verify_checkpoint_health(str(tmp_path / "missing.pt")) is False

        # 2. File below size threshold (< 1 MB)
        tiny_file = tmp_path / "tiny.pt"
        tiny_file.write_bytes(b"0" * 500)
        assert orch.verify_checkpoint_health(str(tiny_file), min_size_mb=1.0) is False

        # 3. Valid checkpoint dictionary with weights (> 1 MB)
        valid_file = tmp_path / "valid.pt"
        payload = {
            "model_state_dict": {
                "dummy_weight": torch.randn(600, 600)  # ~1.44 MB float32 tensor
            },
            "step": 100,
        }
        torch.save(payload, valid_file)
        assert orch.verify_checkpoint_health(str(valid_file), min_size_mb=1.0) is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])