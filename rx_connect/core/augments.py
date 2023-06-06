from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Union

import numpy as np
import torch
import torchshow as ts


@dataclass
class BasicAugTransform(ABC):
    """Abstract class to define the basic structure for image augmentation transformations.

    Any derived class needs to implement the `__call__` method where the actual transformation
    is performed.
    """

    def __init__(self) -> None:
        super().__init__()

    def show_transforms(
        self,
        image: np.ndarray,
        n_transforms: int = 1,
        **kwargs: Any,
    ) -> None:
        """Show different transforms applied to an image."""
        image_t = np.stack([self(image, **kwargs) for _ in range(n_transforms)])
        ts.show(image_t)

    def show_batch_transforms(self, images: Union[np.ndarray, List[np.ndarray]], **kwargs: Any) -> None:
        """Show the transforms to the batch of images."""
        images = [images] if isinstance(images, np.ndarray) else images
        image_t = np.stack([self(image, **kwargs) for image in images])
        ts.show(image_t)

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass
