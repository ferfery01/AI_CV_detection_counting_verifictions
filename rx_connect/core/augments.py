from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
import torchshow as ts
from albumentations.pytorch.transforms import ToTensorV2


@dataclass
class BasicAugTransform(ABC):
    """Abstract class to define the basic structure for image augmentation transformations.

    Any derived class needs to implement the `__call__` method where the actual transformation
    is performed.
    """

    train: bool = True
    """Whether the transform is for training or not.
    """
    normalize: bool = True
    """Whether to normalize images.
    """

    def __init__(self) -> None:
        super().__init__()

    def init_final_transforms(self, normalize: bool) -> A.Compose:
        """Initialize the final transforms to apply. If `normalize` is True, then the final
        transform will include normalization. Otherwise, it will only convert the image to a
        tensor.
        """
        if normalize:
            return A.Compose(
                [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]
            )
        else:
            return ToTensorV2()

    def show_transforms(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        n_transforms: int = 1,
        **kwargs: Any,
    ) -> None:
        """Show different transforms applied to an image and/or mask."""
        if mask is not None:
            image_mask_t = [list(self(image, mask, **kwargs)) for _ in range(n_transforms)]
            ts.show(image_mask_t)
        else:
            image_t = np.stack([self(image, **kwargs) for _ in range(n_transforms)])
            ts.show(image_t)

    def show_batch_transforms(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        masks: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        **kwargs: Any,
    ) -> None:
        """Show the transforms to the batch of images and/or masks."""
        images = [images] if isinstance(images, np.ndarray) else images
        masks = [masks] if isinstance(masks, np.ndarray) else masks

        if masks is not None:
            assert len(images) == len(masks), "Number of images and masks must be the same."
            image_mask_t = [list(self(image, mask, **kwargs)) for image, mask in zip(images, masks)]
            ts.show(image_mask_t)
        else:
            image_t = np.stack([self(image, **kwargs) for image in images])
            ts.show(image_t)

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pass
