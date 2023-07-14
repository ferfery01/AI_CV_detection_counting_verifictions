from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import albumentations as A
import numpy as np
import torch
from albumentations.core.transforms_interface import ImageOnlyTransform, to_tuple
from albumentations.pytorch.transforms import ToTensorV2

from rx_connect.core.augments import BasicAugTransform
from rx_connect.core.transforms_utils import rotate_image


class CustomSigmoidContrast(ImageOnlyTransform):
    """Apply a sigmoid contrast transformation to images."""

    def __init__(
        self,
        gain: Union[int, Tuple[int, int]] = 10,
        cutoff: Union[float, Tuple[float, float]] = 0.5,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super(CustomSigmoidContrast, self).__init__(always_apply, p)
        self.gain = to_tuple(gain)
        self.cutoff = to_tuple(cutoff)

    def apply(self, image: np.ndarray, gain: int, cutoff: float, **params: dict) -> np.ndarray:
        image = (image / 255.0).astype(np.float32)  # normalization to [0.0, 1.0] is required
        image = 1 / (1 + np.exp(gain * (cutoff - image)))  # sigmoid function
        return (image * 255).astype(np.uint8)  # convert back to original range

    def get_params(self) -> Dict[str, float]:
        return {
            "gain": np.random.uniform(self.gain[0], self.gain[1]),
            "cutoff": np.random.uniform(self.cutoff[0], self.cutoff[1]),
        }


@dataclass
class RefConsTransform(BasicAugTransform):
    """A data augmentation transform that applies random affine transformations,
    brightness and contrast adjustments, and Gaussian blur to reference and
    consumer images.
    """

    train: bool = True
    """Whether to apply augmentations during training or evaluation.
    """
    add_perspective: bool = False
    """Whether to add perspective distortion to images.
    """
    normalize: bool = True
    """Whether to normalize images.
    """
    rot_angle: int = 180
    """The maximum angle of rotation to apply to images.
    """
    max_scale: float = 1.2
    """The maximum scale factor to apply to images.
    """
    low_gblur: float = 0.8
    """The minimum sigma of the Gaussian blur kernel to apply to images.
    """
    high_gblur: float = 1.2
    """The maximum sigma of the Gaussian blur kernel to apply to images.
    """
    addgn_base_ref: float = 0.005
    """The base amount of Gaussian noise to add to reference images.
    """
    addgn_base_cons: float = 0.0008
    """The base amount of Gaussian noise to add to contrastive images.
    """

    def __post_init__(self):
        """Initialize all the different data augmentation transforms."""
        self.affine_list = self.init_affine_list()
        self.color_seq = self.init_color_seq()

        self.ref_seq = self.init_ref_seq()
        self.cons_seq = self.init_cons_seq()

        self.final_transforms = self.init_final_transforms()
        self.ref_transforms = self.init_ref_transform()
        self.cons_transforms = self.init_cons_transform()

    def init_affine_list(self) -> List[A.Compose]:
        """Define the affine list for transformations."""
        affine_list = [
            A.Compose(
                [
                    A.Affine(
                        rotate=(-self.rot_angle, self.rot_angle),
                        scale=(0.8, self.max_scale),
                        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                        p=1.0,
                    ),
                    A.Affine(shear=(-4, 4), p=0.5),
                ]
            )
        ]

        if self.add_perspective:
            affine_list += [
                A.Perspective(scale=(0.01, 0.1), p=0.5),
            ]

        return affine_list

    def init_color_seq(self) -> List[A.Compose]:
        """Define the color sequence for transformations."""
        color_seq = [
            A.Compose(
                [
                    A.ImageCompression(quality_lower=70, quality_upper=100),
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.1, 0.1), contrast_limit=(-0.3, 1.3), p=1.0
                    ),
                ]
            ),
            A.Compose(
                [
                    A.ImageCompression(quality_lower=40, quality_upper=100),
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.2, 0.2), contrast_limit=(-0.6, 1.4), p=1.0
                    ),
                ]
            ),
        ]

        if self.add_perspective:
            color_seq += [
                A.RandomGamma(gamma_limit=(60, 200), p=0.1),
                CustomSigmoidContrast(gain=(8, 12), cutoff=(0.2, 0.8), p=0.1),
            ]

        return color_seq

    def init_ref_seq(self):
        """Define the reference sequence for transformations."""
        return A.Compose(
            self.affine_list
            + [
                A.OneOf(self.color_seq),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(0.0, 3 * self.addgn_base_ref * 255), p=0.5),
                        A.GaussNoise(var_limit=(0.0, self.addgn_base_ref * 255), p=0.5),
                    ]
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(sigma_limit=(0, self.high_gblur)),
                        A.GaussianBlur(sigma_limit=(0, self.low_gblur)),
                    ]
                ),
            ],
            p=1.0,
        )

    def init_cons_seq(self):
        """Define the consumer sequence for transformations."""
        return A.Compose(
            self.affine_list
            + [
                A.RandomBrightnessContrast(brightness_limit=(-0.04, 0.04), contrast_limit=(-0.1, 1.1), p=1.0),
                A.GaussNoise(var_limit=(0.0, 5 * self.addgn_base_cons * 255), p=0.5),
                A.GaussianBlur(sigma_limit=(0, self.low_gblur)),
            ],
            p=1.0,
        )

    def init_final_transforms(self) -> A.Compose:
        """Define the final transforms for both reference and consumer images."""
        if self.normalize:
            return A.Compose(
                [
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
        else:
            return ToTensorV2()

    def init_ref_transform(self) -> A.Compose:
        """Define the final transforms for reference images."""
        return A.Compose([self.ref_seq, self.final_transforms])

    def init_cons_transform(self) -> A.Compose:
        """Define the final transforms for consumer images."""
        return A.Compose([self.cons_seq, self.final_transforms])

    def finetune_transform(self, is_ref: bool) -> A.Compose:
        """Define the final transforms depending on whether the image is a reference or
        consumer image and whether the model is in training mode or not."""
        if self.train:
            return self.cons_transforms if not is_ref else self.ref_transforms
        else:
            return self.final_transforms

    def __call__(
        self,
        image: np.ndarray,
        is_ref: bool = True,
        rot_degree: int = 0,
    ) -> torch.Tensor:
        """Apply the transforms to the image.

        Args:
            image: The image to apply the transforms to.
            is_ref: Whether the image is a reference image or not.
            rot_degree: The rotation angle in degrees in counter-clockwise direction.

        Returns:
            The transformed image as a torch tensor.
        """
        # Rotate the image
        image = rotate_image(image, angle=rot_degree)

        return self.finetune_transform(is_ref)(image=image)["image"]
