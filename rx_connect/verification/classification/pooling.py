from typing import Optional

import torch
import torch.nn as nn


class GlobalAvgPool(nn.Module):
    """Global Average Pooling layer.

    Used widely in networks like ResNet, Inception, DenseNet, etc. This layer is
    usually applied to the output of the feature extractor to reduce the dimension
    of the tensor before applying the classifier head.

    If `dim_reduction` is not None, then a 1x1 convolutional layer is applied before
    the pooling layer to reduce the dimensionality to `dim_reduction`.

    Args:
        input_dim (int): Input dimension of the tensor. This is usually the output
            number of channels from the feature extractor.
        dim_reduction (Optional[int]): Dimension reduction to be applied to the input
            tensor during pooling.

    Returns:
        torch.Tensor: Output tensor after applying pooling.
    """

    output_dim: int

    def __init__(self, input_dim: int = 2048, dim_reduction: Optional[int] = None) -> None:
        super().__init__()
        self.dr = dim_reduction

        # Apply dimensionality reduction, if `dim_reduction` is not None
        self.dr_block = None
        if self.dr is not None and input_dim != self.dr:
            self.dr_block = nn.Sequential(
                nn.Conv2d(input_dim, self.dr, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.dr),
                nn.ReLU(inplace=True),
            )

        self.output_dim = self.dr if self.dr is not None else input_dim
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GlobalAvgPool layer."""
        if self.dr_block is not None:
            x = self.dr_block(x)
        return self.avgpool(x)
