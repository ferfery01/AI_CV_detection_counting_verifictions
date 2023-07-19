from typing import Optional, TypedDict

import torch


class MultiHeadModelOutput(TypedDict):
    emb: torch.Tensor
    logits: torch.Tensor
    arcface_logits: Optional[torch.Tensor]


class LossWeights(TypedDict):
    cls: float
    arcface: float


class MultiheadLossType(TypedDict):
    cls: Optional[torch.Tensor]
    arcface: Optional[torch.Tensor]
    total: torch.Tensor
