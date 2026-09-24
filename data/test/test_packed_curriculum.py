"""
data/test/test_packed_curriculum.py: Unit tests for physical packing.
"""

import glob
import os
import numpy as np
import pytest

from data.prepare_packed_curriculum import LocalBinReader, pack_curriculum_to_shards
from tokenizer.factory import get_tokenizer


@pytest.fixture
def mock_upstream_sources(tmp_path):
    tokenizer = get_tokenizer("tiktoken")
    eot = tokenizer.eot_id

    # Create 3 distinct mock binary sources
    s1_dir = tmp_path / "source_general"
    s2_dir = tmp_path / "source_code"
    s3_dir = tmp_path / "source_math"
    for d in (s1_dir, s2_dir, s3_dir):
        d.mkdir()

    # Source 1: tokens in 1000..1999 (standard uint16 token serialization)
    doc1 = np.array([1001, 1002, 1003, eot, 1004, 1005, eot], dtype=np.uint16)
    doc1.tofile(str(s1_dir / "shard_00.bin"))

    # Source 2: tokens in 2000..2999
    doc2 = np.array([2001, 2002, eot, 2003, 2004, 2005, eot], dtype=np.uint16)
    doc2.tofile(str(s2_dir / "shard_00.bin"))

    # Source 3: tokens in 3000..3999
    doc3 = np.array([3001, 3002, 3003, 3004, eot], dtype=np.uint16)
    doc3.tofile(str(s3_dir / "shard_00.bin"))

    readers = [
        LocalBinReader("general", str(s1_dir), weight=0.60, eot_token_id=eot, dtype=np.uint16),
        LocalBinReader("code", str(s2_dir), weight=0.20, eot_token_id=eot, dtype=np.uint16),
        LocalBinReader("math", str(s3_dir), weight=0.20, eot_token_id=eot, dtype=np.uint16),
    ]
    return readers, str(tmp_path / "output_packed")


def test_packed_curriculum_execution_and_ratios(mock_upstream_sources):
    readers, out_dir = mock_upstream_sources
    budget = 10_000
    shard_size = 2_500

    pack_curriculum_to_shards(
        sources=readers,
        output_dir=out_dir,
        total_token_budget=budget,
        shard_size_tokens=shard_size,
    )

    # Shards are written using the train_*.bin naming contract
    shards = sorted(glob.glob(os.path.join(out_dir, "train_*.bin")))
    assert len(shards) == 4  # 10,000 / 2,500 = 4 shards

    # Verified against uint16 shard encoding
    all_tokens = np.concatenate([np.fromfile(s, dtype=np.uint16) for s in shards])
    assert len(all_tokens) == budget

    # Count source frequencies
    c_gen = np.sum((all_tokens >= 1000) & (all_tokens < 2000))
    c_code = np.sum((all_tokens >= 2000) & (all_tokens < 3000))
    c_math = np.sum((all_tokens >= 3000) & (all_tokens < 4000))

    total_valid = c_gen + c_code + c_math
    assert total_valid > 0

    # Verify distribution matches 60/20/20 within tolerance
    p_gen = c_gen / total_valid
    p_code = c_code / total_valid
    p_math = c_math / total_valid

    assert abs(p_gen - 0.60) < 0.08
    assert abs(p_code - 0.20) < 0.08
    assert abs(p_math - 0.20) < 0.08