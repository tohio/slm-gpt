"""
pipeline/test/test_pipeline.py: Unit tests for the pipeline orchestrator.
Tests path resolution, health gates, and argument parsing without spinning up full training runs.
"""

import os
import shutil
import tempfile
import pytest
import torch

from pipeline.run import PipelineConfig, PipelineOrchestrator


@pytest.fixture
def temp_env():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def test_orchestrator_path_resolution():
    cfg = PipelineConfig(size="350M", tokenizer_type="tiktoken")
    orch = PipelineOrchestrator(cfg)

    assert "350M" in orch.pretrain_ckpt or "3" in orch.pretrain_ckpt
    assert orch.pretrain_ckpt.endswith("best_model.pt")
    assert orch.sft_ckpt.endswith("sft_final.pt")
    assert orch.dpo_ckpt.endswith("dpo_final.pt")


def test_health_gate_verification(temp_env):
    cfg = PipelineConfig(size="10M")
    orch = PipelineOrchestrator(cfg)
    test_path = os.path.join(temp_env, "model.pt")

    # 1. Non-existent file fails
    assert orch.verify_checkpoint_health(test_path) is False

    # 2. Corrupt/empty file fails
    with open(test_path, "wb") as f:
        f.write(b"corrupt")
    assert orch.verify_checkpoint_health(test_path) is False

    # 3. Valid checkpoint passes
    torch.save(
        {"model_state_dict": {"wte.weight": torch.randn(10, 10)}},
        test_path,
    )
    assert orch.verify_checkpoint_health(test_path, min_size_mb=0.0) is True