from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rx_connect.core.types.verification.model import (
    LossWeights,
    MultiheadLossType,
    MultiHeadModelOutput,
)


class MultiHeadLoss(nn.Module):
    """Module for computing the multi-head loss."""

    def __init__(self, n_classes: int, use_side_labels: bool, weights: LossWeights) -> None:
        super().__init__()
        self.n_classes = n_classes  # classification-labels need to be shifted for back-sides
        self.use_side_labels = use_side_labels
        self.weights = weights

    def forward(
        self,
        outputs: MultiHeadModelOutput,
        target: torch.Tensor,
        is_front: Optional[torch.Tensor] = None,
        is_ref: Optional[torch.Tensor] = None,
    ) -> MultiheadLossType:
        if not self.use_side_labels:
            is_front = None

        device = outputs["emb"].device
        weighted_loss = torch.zeros(1, dtype=torch.float).to(device)

        # Shift the target labels for back-side pills
        if is_front is not None:
            target = target.clone().detach()
            target[~(is_front.bool())] += self.n_classes

        # Compute the classification (Softmax) loss
        cls_loss: Optional[torch.Tensor] = None
        if self.weights["cls"] > 0:
            cls_loss = F.cross_entropy(outputs["cls_logits"], target, reduction="mean")
            weighted_loss += cls_loss * self.weights["cls"]

        # Compute the arcface loss
        acrface_loss: Optional[torch.Tensor] = None
        if self.weights["angular"] > 0:
            assert outputs["angular_logits"] is not None, "Angular logits must be provided."
            acrface_loss = F.cross_entropy(outputs["angular_logits"], target, reduction="mean")
            weighted_loss += acrface_loss * self.weights["angular"]

        return MultiheadLossType(cls=cls_loss, arcface=acrface_loss, total=weighted_loss)
