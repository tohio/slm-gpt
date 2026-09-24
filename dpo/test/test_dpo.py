"""dpo/test/test_dpo.py: Unit tests for DPO loss engine, log-prob calculation, and collator."""

import pytest
import torch
from dpo.collator import DPODataCollator
from dpo.dataset import IGNORE_INDEX
from dpo.loss import DPOLoss, get_batch_logps


def test_dpo_logps_masking_invariance():
    """Verifies that prompt tokens marked with IGNORE_INDEX (-100) have zero
    effect on sequence log-probabilities.
    """
    B, T, V = 2, 6, 128
    logits = torch.randn(B, T, V)

    # Sequence where positions 0, 1 are prompt, positions 2, 3, 4 are completion, position 5 is pad
    labels_with_mask = torch.tensor(
        [
            [IGNORE_INDEX, IGNORE_INDEX, 14, 55, 92, IGNORE_INDEX],
            [IGNORE_INDEX, IGNORE_INDEX, 8, 22, 105, IGNORE_INDEX],
        ],
        dtype=torch.long,
    )

    logps = get_batch_logps(logits, labels_with_mask, ignore_index=IGNORE_INDEX)

    assert logps.shape == (B,)
    assert not torch.isnan(logps).any(), "Log probabilities contain NaN."
    assert not torch.isinf(logps).any(), "Log probabilities contain Inf."

    # Invariance check: Modifying prompt logits should NOT alter logps of the completion
    modified_logits = logits.clone()
    modified_logits[:, :2, :] += 100.0  # Shift prompt logits drastically

    modified_logps = get_batch_logps(
        modified_logits, labels_with_mask, ignore_index=IGNORE_INDEX
    )
    diff = (logps - modified_logps).abs().max().item()
    assert (
        diff < 1e-5
    ), f"Prompt tokens leaked into completion log-probs! Discrepancy: {diff}"


def test_dpo_mathematical_objective():
    """Verifies DPO loss calculation, implicit rewards, and preference margin."""
    beta = 0.1
    criterion = DPOLoss(beta=beta)

    # Case 1: Policy strongly favors chosen over rejected
    # log pi(chosen) = -1.0, log pi(rejected) = -8.0
    # log ref(chosen) = -4.0, log ref(rejected) = -4.0
    pi_c = torch.tensor([-1.0, -1.2])
    pi_r = torch.tensor([-8.0, -7.8])
    ref_c = torch.tensor([-4.0, -4.0])
    ref_r = torch.tensor([-4.0, -4.0])

    loss, metrics = criterion(pi_c, pi_r, ref_c, ref_r)

    # Implicit reward: r(x, y) = beta * (log pi - log ref)
    # chosen_reward = 0.1 * (-1.0 - (-4.0)) = +0.3
    # rejected_reward = 0.1 * (-8.0 - (-4.0)) = -0.4
    # margin = +0.7 -> -log(sigmoid(0.7)) ≈ 0.4099
    assert metrics["accuracy"].item() == 1.0
    assert metrics["reward_margin"].item() > 0.5
    assert metrics["chosen_rewards"].item() > 0.0
    assert metrics["rejected_rewards"].item() < 0.0
    assert (
        loss.item() < 0.5
    ), f"Loss should be under 0.5 for aligned preference: {loss.item()}"

    # Case 2: Policy favors rejected (negative margin -> high loss)
    loss_bad, metrics_bad = criterion(pi_r, pi_c, ref_c, ref_r)
    assert metrics_bad["accuracy"].item() == 0.0
    assert metrics_bad["reward_margin"].item() < 0.0
    assert loss_bad.item() > loss.item()


def test_dpo_collator_dual_padding():
    """Verifies that chosen and rejected sequences are padded to matching lengths
    rounded to multiples of 16.
    """
    collator = DPODataCollator(pad_token_id=50256, pad_to_multiple_of=16)

    features = [
        {
            "chosen_input_ids": [10, 20, 30, 40],
            "chosen_labels": [IGNORE_INDEX, IGNORE_INDEX, 30, 40],
            "rejected_input_ids": [10, 20, 99],
            "rejected_labels": [IGNORE_INDEX, IGNORE_INDEX, 99],
        },
        {
            "chosen_input_ids": [10, 20, 31, 41, 51, 61, 71],
            "chosen_labels": [
                IGNORE_INDEX,
                IGNORE_INDEX,
                31,
                41,
                51,
                61,
                71,
            ],
            "rejected_input_ids": [10, 20, 91, 92],
            "rejected_labels": [IGNORE_INDEX, IGNORE_INDEX, 91, 92],
        },
    ]

    batch = collator(features)

    # Max sequence length across all chosen & rejected is 7.
    # Rounded to next multiple of 16 -> 16
    assert batch["chosen_input_ids"].shape == (2, 16)
    assert batch["chosen_labels"].shape == (2, 16)
    assert batch["rejected_input_ids"].shape == (2, 16)
    assert batch["rejected_labels"].shape == (2, 16)

    # Verify padding fill
    assert (batch["chosen_input_ids"][0, 4:] == 50256).all()
    assert (batch["chosen_labels"][0, 4:] == IGNORE_INDEX).all()
    assert (batch["rejected_input_ids"][0, 3:] == 50256).all()
    assert (batch["rejected_labels"][0, 3:] == IGNORE_INDEX).all()