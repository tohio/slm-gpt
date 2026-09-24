"""
dpo/loss.py: Stable sequence-level log-probability computation and DPO objective.
Casts logits to float32 to prevent numerical underflow during log-ratio subtractions.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .dataset import IGNORE_INDEX
except ImportError:
    from dpo.dataset import IGNORE_INDEX


def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """
    Computes the sum of log probabilities per sequence over completion tokens.

    Args:
        logits: Shape (B, T, V)
        labels: Shape (B, T) with prompt tokens masked to `ignore_index`

    Returns:
        logps: Shape (B,) sum of log probabilities for supervised response tokens.
    """
    # Shift for causal prediction: logits at t predict label at t+1
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()

    loss_mask = shift_labels != ignore_index

    # Negative cross entropy equals log P(y_t | y_<t, x)
    log_probs = -F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).view(shift_labels.shape)

    return (log_probs * loss_mask).sum(dim=-1)


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization loss module (Rafailov et al., 2023).
    Computes the implicit reward formulation r(x, y) = beta * log(pi / pi_ref).
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Log ratios: log (pi_theta / pi_ref)
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = self.beta * (pi_logratios - ref_logratios)

        # Objective: -E[log sigma(beta * (pi_ratio - ref_ratio))]
        loss = -F.logsigmoid(logits).mean()

        # Track implicit scalar rewards for chosen and rejected completions
        chosen_rewards = (self.beta * (policy_chosen_logps - reference_chosen_logps)).detach()
        rejected_rewards = (self.beta * (policy_rejected_logps - reference_rejected_logps)).detach()
        reward_margin = (chosen_rewards - rejected_rewards).mean()
        reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()

        metrics = {
            "loss": loss.detach(),
            "chosen_rewards": chosen_rewards.mean(),
            "rejected_rewards": rejected_rewards.mean(),
            "reward_margin": reward_margin,
            "accuracy": reward_accuracy,
        }

        return loss, metrics