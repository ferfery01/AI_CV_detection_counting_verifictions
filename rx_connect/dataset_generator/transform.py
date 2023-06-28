from typing import Optional, Tuple

import albumentations as A
import numpy as np


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


def resize_and_transform_pill(
    image: np.ndarray,
    mask: np.ndarray,
    longest_min: int = 224,
    longest_max: int = 224,
    allow_defects: bool = False,
    augmentations: Optional[A.BasicTransform] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize the pill image and the corresponding mask to the given height and width.
    Also, apply some random augmentations to the pill image.

    Args:
        image: pill image as a numpy array.
        mask: binary mask of the pill image.
        longest_min: minimum size of the longest side of the resized image.
        longest_max: maximum size of the longest side of the resized image.
        allow_defects: whether to allow defects in the pill image.
        augmentations: augmentations to apply to the pill image.

    Returns:
        tuple of transformed pill image and mask.
    """
    height, width = image.shape[:2]

    long_side = max(height, width)
    short_side = min(height, width)

    # Randomly select the long side of the new image. The short side will be resized to
    # keep the aspect ratio of the original image.
    long_new = np.random.randint(longest_min, longest_max + 1)
    short_new = int(short_side * long_new / long_side)
    h_new, w_new = (long_new, short_new) if height > width else (short_new, long_new)

    # Resize the image to the provided size.
    transform_resize = A.Resize(height=h_new, width=w_new)
    transform_resized = transform_resize(image=image, mask=mask)
    img_t, mask_t = transform_resized["image"], transform_resized["mask"]

    # Initialize the augmentations if not provided.
    augmentations = augmentations or A.Compose(
        [
            A.Rotate(limit=90, border_mode=0, mask_value=0, p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.02,
                contrast_limit=0.02,
                brightness_by_max=True,
            ),
        ]
    )

    # Add random crop to the augmentations if defects are allowed.
    if allow_defects:
        # Randomly select the fraction of the image to crop. The fraction is selected
        # from the range [80, 100) percent.
        fraction = 0.01 * np.random.randint(80, 100)

        # Apply random crop augmentation 25% of the time
        augmentations = A.Compose(
            [
                augmentations,
                A.RandomCrop(p=0.25, height=int(fraction * h_new), width=int(fraction * w_new)),
            ]
        )

    # Apply the augmentations to the image.
    transforms_aug = augmentations(image=img_t, mask=mask_t)
    img_t, mask_t = transforms_aug["image"], transforms_aug["mask"]

    return img_t, mask_t
