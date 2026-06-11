from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricFocalBinaryLoss(nn.Module):
    def __init__(self, alpha: float = 0.65, gamma: float = 1.5, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probabilities = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        pos = -self.alpha * targets * ((1.0 - probabilities) ** self.gamma) * torch.log(probabilities)
        neg = -(1.0 - self.alpha) * (1.0 - targets) * (probabilities ** self.gamma) * torch.log(1.0 - probabilities)
        loss = pos + neg
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def consistency_loss_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor, kind: str = "mse") -> torch.Tensor:
    if kind == "kl":
        probs_a = torch.sigmoid(logits_a)
        probs_b = torch.sigmoid(logits_b)
        dist_a = torch.stack([1.0 - probs_a, probs_a], dim=1).clamp(1e-6, 1.0 - 1e-6)
        dist_b = torch.stack([1.0 - probs_b, probs_b], dim=1).clamp(1e-6, 1.0 - 1e-6)
        return F.kl_div(dist_b.log(), dist_a, reduction="batchmean")
    return F.mse_loss(logits_a, logits_b)
