from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union, overload

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
    """Whether the transform is for training or validation/testing.
    """
    normalize: bool = True
    """Whether to normalize the image.
    """

    def init_final_transforms(self, normalize: bool) -> A.Compose:
        """Initialize the final transforms to apply. If `normalize` is True, then the final
        transform will include normalization. Otherwise, it will only convert the image to a
        tensor.
        """
        tfms = (
            [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]
            if normalize
            else [ToTensorV2()]
        )
        return A.Compose(tfms)

    def show_transforms(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        n_transforms: int = 1,
        **kwargs: Any,
    ) -> None:
        """Show different transforms applied to an image and/or mask."""
        if mask is not None:
            # To visualize the pair of image and mask properly, the mask is first converted to
            # a 3-channel image and is then concatenated to the image along the last axis.
            # For better visualization, we iterate over the number of transforms and show the results
            for _ in range(n_transforms):
                image_t, mask_t = self(image, mask, **kwargs)
                ts.show([image_t, mask_t])
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
            for image, mask in zip(images, masks):
                image_t, mask_t = self(image, mask, **kwargs)
                ts.show([image_t, mask_t])
        else:
            image_t = np.stack([self(image, **kwargs) for image in images])
            ts.show(image_t)

    @overload
    @abstractmethod
    def __call__(self, image: np.ndarray, mask: None = None, *args: Any, **kwargs: Any) -> torch.Tensor:
        ...

    @overload
    @abstractmethod
    def __call__(
        self, image: np.ndarray, mask: np.ndarray, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @abstractmethod
    def __call__(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None, *args: Any, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Perform the transformation.

        Args:
            image: The image to be transformed.
            mask: Optional mask to be transformed.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple of (transformed_image, transformed_mask) if mask is provided,
            else just the transformed_image.
        """
        raise NotImplementedError
