import random
from typing import Optional, Tuple, Union

import albumentations as A
import numpy as np
from skimage.transform import rescale

from rx_connect.core.images.transforms import resize_to_square
from rx_connect.core.types.generator import PillMask
from rx_connect.core.utils.func_utils import to_tuple

COLOR_TRANSFORMS = [
    A.CLAHE(),
    A.RandomBrightnessContrast(),
    A.RandomGamma(),
    A.MultiplicativeNoise(multiplier=(0.8, 1.2)),
    A.RandomToneCurve(scale=0.2),
]
"""Albumentations composition of color transforms."""

NOISE_TRANSFORMS = [
    A.AdvancedBlur(),
    A.MedianBlur(),
    A.ZoomBlur(max_factor=1.05, p=0.25),
    A.GaussNoise(),
    A.ImageCompression(quality_lower=50),
    A.ISONoise(),
    A.PixelDropout(dropout_prob=0.05),
    A.Emboss(alpha=(0.8, 1.0), strength=(0.7, 1.0)),
    A.Sharpen(),
    # A.Spatter(mode=["rain", "mud"], p=0.25),
]
"""Albumentations composition of noise transforms."""


def resize_bg(img: np.ndarray, desired_max: int = 1920, desired_min: Optional[int] = None) -> np.ndarray:
    """Resize the background image to the given height and width. Some images might
    be horizontal, some might be vertical. This function will resize the image to have
    the long side equal to `desired_max` and the short side equal to `desired_min` if
    provided, otherwise the short side will be resized to keep the aspect ratio of the
    original image.

    Args:
        img (np.ndarray): background image.
        desired_max (int): desired maximum of the output. Defaults to 1920.
        desired_min (int): desired minimum of the output. Defaults to None.

    Returns:
        np.ndarray: resized image.
    """
    height, width = img.shape[:2]

    long_side = max(height, width)
    short_side = min(height, width)

    # Resize the image so that the long side is equal to `desired_max` and the short side
    # is equal to `desired_min` if provided, otherwise the short side will be resized to
    # keep the aspect ratio of the original image.
    long_new = desired_max
    short_new = int(short_side * long_new / long_side) if desired_min is None else desired_min
    h_new, w_new = (long_new, short_new) if height > width else (short_new, long_new)

    # Resize the image to the new size.
    transform_resize = A.Resize(height=h_new, width=w_new)
    img = transform_resize(image=img)["image"]

    return img


def rescale_pill_and_mask(
    image: np.ndarray, mask: np.ndarray, scale: Union[float, Tuple[float, float]] = 1.0
) -> PillMask:
    """Rescale the pill image and the corresponding mask by the given scale.

    Args:
        image (np.ndarray): pill image as a numpy array.
        mask (np.ndarray): binary mask of the pill image.
        scale (Union[float, Tuple[float, float]], optional): The scaling factor to use for rescaling
            the pill image and mask. If a tuple is provided, then the scaling factor is randomly
            sampled from the range (min, max). If a float is provided, then the scaling factor is fixed.

    Returns:
        PillMask: rescaled pill image and mask.
    """
    # Randomly sample the scaling factor from the given range
    s = random.uniform(*to_tuple(scale))

    # Scale image and mask by the given scale
    image_t = rescale(image, (s, s), mode="constant", order=1, anti_aliasing=True, channel_axis=2)
    mask_t = rescale(mask, (s, s), mode="constant", order=0)

    # Resize the image and mask to a square
    image_t = resize_to_square(image_t)
    mask_t = resize_to_square(mask_t)

    return PillMask(image_t, mask_t)


def transform_pill(
    image: np.ndarray,
    mask: np.ndarray,
    allow_defects: bool = False,
    augmentations: Optional[A.BasicTransform] = None,
) -> PillMask:
    """Apply random color and/or noise augmentations to the pill image.

    Args:
        image: pill image as a numpy array.
        mask: binary mask of the pill image.
        allow_defects: whether to allow defects in the pill image.
        augmentations: color and/or noise augmentations to apply to the pill image.

    Returns:
        transformed pill image and mask.
    """
    # Convert the image and the boolean mask to uint8.
    image = (
        (image * 255).astype(np.uint8) if image.dtype in (np.float32, np.float64) else image.astype(np.uint8)
    )
    mask = mask.astype(np.uint8)

    # Initialize the transforms to resize the image and mask.
    geometric_aug = A.Compose([A.Rotate(limit=180, border_mode=0, mask_value=0, p=1.0)])
    color_aug = augmentations or A.Compose(
        [A.RandomBrightnessContrast(brightness_limit=0.02, contrast_limit=0.02, brightness_by_max=True)]
    )

    # Add random crop to the augmentations if defects are allowed.
    if allow_defects:
        height, width = mask.shape
        # Randomly select the fraction of the image to crop. The fraction is selected
        # from the range [80, 100) percent.
        fraction = 0.01 * np.random.randint(80, 100)

        # Apply random crop augmentation 25% of the time
        geometric_aug = A.Compose(
            [
                geometric_aug,
                A.RandomCrop(p=0.25, height=int(fraction * height), width=int(fraction * width)),
            ]
        )

    # First, apply the geometric augmentation to the image and mask.
    transforms_rot = geometric_aug(image=image, mask=mask)
    image_t, mask_t = transforms_rot["image"], transforms_rot["mask"]

    # Second, apply the color augmentation to the image.
    image_t = color_aug(image=image_t)["image"]

    return PillMask(image_t, mask_t)


def apply_augmentations(image: np.ndarray, apply_color: bool = True, apply_noise: bool = True) -> np.ndarray:
    """Apply random color and/or noise augmentations to the image.

    The augmentations are applied with a 50% probability. If `apply_color` and `apply_noise`
    are both False, the original image is returned. If only one of them is True, only the
    corresponding augmentations are applied. If both are True, both augmentations are applied after one
    another. Only one augmentations from each group is applied at a time.
    """
    transforms = []
    if apply_color:
        transforms.append(A.OneOf(COLOR_TRANSFORMS))
    if apply_noise:
        transforms.append(A.OneOf(NOISE_TRANSFORMS))

    if len(transforms) > 0:
        return A.Compose(transforms)(image=image)["image"]
    else:
        return image
