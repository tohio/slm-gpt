"""
pretrain/test/test_pretrain.py: Automated tests for the pre-training engine.
Tests sharded dataloading, distributed rank partitioning, optimization, and checkpointing.
"""

import math
import os
import shutil
import tempfile
import numpy as np
import pytest
import torch

from pretrain.train import (
    DistributedShardedDataLoader,
    configure_optimizers,
    estimate_loss,
    get_lr,
)
from scaling.config import RuntimeConfig
from transformer import DecoderOnlyTransformer


@pytest.fixture
def temp_bin_dataset():
    """Creates a temporary synthetic binary dataset for pretrain testing."""
    tmpdir = tempfile.mkdtemp()
    seq_len = 64
    total_tokens = 5000

    tokens = np.arange(total_tokens, dtype=np.uint16)
    train_path = os.path.join(tmpdir, "train.bin")
    val_path = os.path.join(tmpdir, "val.bin")

    tokens.tofile(train_path)
    tokens[:500].tofile(val_path)

    yield tmpdir, seq_len
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_distributed_sharded_dataloader_disjoint(temp_bin_dataset):
    tmpdir, seq_len = temp_bin_dataset
    batch_size = 2

    # Simulate Rank 0 and Rank 1 in a 2-GPU cluster
    loader_r0 = DistributedShardedDataLoader(
        tmpdir, "train", batch_size=batch_size, seq_len=seq_len, rank=0, world_size=2, device=torch.device("cpu")
    )
    loader_r1 = DistributedShardedDataLoader(
        tmpdir, "train", batch_size=batch_size, seq_len=seq_len, rank=1, world_size=2, device=torch.device("cpu")
    )

    x0, y0 = loader_r0.next_batch()
    x1, y1 = loader_r1.next_batch()

    # Targets must be inputs shifted by 1
    assert torch.equal(x0[:, 1:], y0[:, :-1])

    # Rank 0 and Rank 1 must receive completely disjoint token streams
    tokens_r0 = set(x0.flatten().tolist())
    tokens_r1 = set(x1.flatten().tolist())
    assert len(tokens_r0.intersection(tokens_r1)) == 0, "Ranks 0 and 1 processed overlapping tokens!"


def test_pretrain_optimization_step(temp_bin_dataset):
    tmpdir, seq_len = temp_bin_dataset

    # Create a small dynamic config for rapid testing
    cfg = RuntimeConfig.build(
        size="10M",
        vocab_size=1024,
        max_seq_len=seq_len,
        micro_batch_size=2,
        global_batch_size=2,
    )

    model = DecoderOnlyTransformer(cfg.model_cfg)
    optimizer = configure_optimizers(model, weight_decay=0.01, learning_rate=1e-3)

    loader = DistributedShardedDataLoader(
        tmpdir, "train", batch_size=cfg.micro_batch_size, seq_len=seq_len, rank=0, world_size=1, device=torch.device("cpu")
    )

    # Initial weights snapshot
    initial_wte = model.wte.weight.clone()

    x, y = loader.next_batch()
    # Clamp to test vocab
    x = x % cfg.model_cfg.vocab_size
    y = y % cfg.model_cfg.vocab_size

    _, loss, _ = model(x, targets=y)
    assert not torch.isnan(loss) and loss.item() > 0.0

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Weights must update
    assert not torch.equal(initial_wte, model.wte.weight)

    # Invariant: Weight tying must remain intact after optimizer update
    assert model.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_estimate_loss_evaluation(temp_bin_dataset):
    tmpdir, seq_len = temp_bin_dataset
    cfg = RuntimeConfig.build(size="10M", vocab_size=1024, max_seq_len=seq_len)
    model = DecoderOnlyTransformer(cfg.model_cfg)

    val_loader = DistributedShardedDataLoader(
        tmpdir, "val", batch_size=2, seq_len=seq_len, rank=0, world_size=1, device=torch.device("cpu")
    )

    val_loss = estimate_loss(model, val_loader, dtype=torch.float32, eval_iters=2)
    assert isinstance(val_loss, float)
    assert not math.isnan(val_loss) and val_loss > 0.0


def test_learning_rate_warmup_and_decay():
    warmup_steps = 100
    max_steps = 1000
    max_lr = 6e-4
    min_lr = 6e-5

    # Step 0 should be warm-up start
    lr_0 = get_lr(0, warmup_steps, max_steps, max_lr, min_lr)
    assert lr_0 == pytest.approx(max_lr * (1 / warmup_steps))

    # At warmup_steps, should be at max_lr
    lr_warmup = get_lr(warmup_steps, warmup_steps, max_steps, max_lr, min_lr)
    assert lr_warmup == pytest.approx(max_lr, rel=1e-2)

    # Beyond max_steps, should floor at min_lr
    lr_end = get_lr(max_steps + 10, warmup_steps, max_steps, max_lr, min_lr)
    assert lr_end == min_lr