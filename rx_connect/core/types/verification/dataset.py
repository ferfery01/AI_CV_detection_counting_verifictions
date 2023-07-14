from typing import List, TypedDict

import torch

"""Dataset Types for each Module.""" ""


class ePillIDDataset(TypedDict):
    image: torch.Tensor
    label: int
    image_name: str
    is_ref: bool
    is_front: bool


class ePillIDDatasetBatch(TypedDict):
    image: torch.Tensor
    label: torch.Tensor
    image_name: List[str]
    is_ref: torch.Tensor
    is_front: torch.Tensor
