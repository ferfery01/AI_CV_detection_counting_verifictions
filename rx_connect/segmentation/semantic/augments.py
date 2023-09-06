from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, overload

import albumentations as A
import numpy as np
import torch

from rx_connect.core.augments import BasicAugTransform


@dataclass
class SegmentTransform(BasicAugTransform):
    """Define the data augmentation transforms for segmentation tasks.

    This class encompasses various spatial and non-spatial transforms, which can be applied to both images
    and masks. The spatial transforms include random cropping, horizontal and vertical flipping, elastic
    transformations, and distortions. The non-spatial transforms include CLAHE, random brightness /contrast
    adjustments, and gamma corrections.

    Usage:
        >> segment_transform = SegmentTransform()
        >> image_transformed, mask_transformed = segment_transform(image, mask)
    """

    image_size: Tuple[int, int] = (405, 720)
    """The target image size to resize to. Format: (height, width)
    """
    pad_divisor: int = 16
    """Ensures that the image dimensions are divisible by this number.
    """

    def __post_init__(self):
        """Initialize all the different data augmentation transforms.
        Includes spatial transforms, color adjustments, and final resize transform.
        """
        self.height, self.width = self.image_size

        self.spatial_tfm = self.init_spatial_transform()
        self.color_tfm = self.init_color_transform()
        self.resize_tfm = A.Resize(self.height, self.width, always_apply=True)
        """The final resize transform to apply to both images and masks.
        """

        self.final_tfms = self.init_final_transforms(normalize=self.normalize)
        self.transforms = self.finetune_transform()

    def init_spatial_transform(self) -> A.Compose:
        """Define the spatial transforms for augmentation."""
        spatial_tfms = A.Compose(
            [
                A.Resize(self.height, self.width, always_apply=True),
                A.RandomSizedCrop(
                    min_max_height=(int(0.8 * self.height), self.height),
                    height=self.height,
                    width=self.width,
                    w2h_ratio=self.width / self.height,
                    p=0.5,
                ),
                A.PadIfNeeded(
                    min_height=None,
                    min_width=None,
                    pad_height_divisor=self.pad_divisor,
                    pad_width_divisor=self.pad_divisor,
                    always_apply=True,
                ),
                A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=0.5),
                A.OneOf(
                    [
                        A.ElasticTransform(alpha=10, sigma=50, alpha_affine=50, p=0.5),
                        A.GridDistortion(distort_limit=0.2, p=0.5),
                        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
                    ],
                    p=0.5,
                ),
            ]
        )
        return spatial_tfms

    def init_color_transform(self) -> A.Compose:
        """Define the non-spatial transforms for augmentation."""
        color_seq = A.Compose(
            [
                A.CLAHE(clip_limit=8, tile_grid_size=(16, 16), p=0.25),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(gamma_limit=(60, 140), p=0.25),
                A.Emboss(p=0.25),
                A.Blur(blur_limit=7, p=0.25),
                A.GaussNoise(p=0.25),
            ]
        )

        return color_seq

    def finetune_transform(self) -> A.Compose:
        """Combine all the transforms together."""
        return A.Compose(
            [self.spatial_tfm, self.color_tfm, self.resize_tfm, self.final_tfms]
            if self.train
            else [self.resize_tfm, self.final_tfms]
        )

    @overload
    def __call__(self, image: np.ndarray, mask: None = None, *args: Any, **kwargs: Any) -> torch.Tensor:
        ...

    @overload
    def __call__(
        self, image: np.ndarray, mask: np.ndarray, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def __call__(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None, *args: Any, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply the transforms to the image and mask."""
        augmented = (
            self.transforms(image=image, mask=mask) if mask is not None else self.transforms(image=image)
        )
        image_t, mask_t = augmented["image"], augmented.get("mask", None)

        return (image_t, mask_t) if mask_t is not None else image_t
