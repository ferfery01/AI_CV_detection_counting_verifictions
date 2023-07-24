from typing import Optional, TypedDict

import torch


class MultiHeadModelOutput(TypedDict):
    emb: torch.Tensor
    cls_logits: torch.Tensor
    angular_logits: Optional[torch.Tensor]


class LossWeights(TypedDict):
    cls: float
    angular: float


class MultiheadLossType(TypedDict):
    cls: Optional[torch.Tensor]
    arcface: Optional[torch.Tensor]
    total: torch.Tensor
