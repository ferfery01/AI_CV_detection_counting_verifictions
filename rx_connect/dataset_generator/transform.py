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
    img: np.ndarray,
    mask: np.ndarray,
    longest_max: int = 224,
    longest_min: int = 224,
    augmentations: Optional[A.BasicTransform] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize the pill image and the corresponding mask to the given height and width.
    Also, apply some random augmentations to the pill image.
    """
    height, width = img.shape[:2]

    long_side = max(height, width)
    short_side = min(height, width)

    # Randomly select the long side of the new image. The short side will be resized to
    # keep the aspect ratio of the original image.
    long_new = np.random.randint(longest_min, longest_max + 1)
    short_new = int(short_side * long_new / long_side)
    h_new, w_new = (long_new, short_new) if height > width else (short_new, long_new)

    # Resize the image to the new size.
    transform_resize = A.Resize(height=h_new, width=w_new)
    transform_resized = transform_resize(image=img, mask=mask)
    img_t, mask_t = transform_resized["image"], transform_resized["mask"]

    # Apply some random augmentations to the pill image.
    augmentations = augmentations or A.Compose(
        [
            A.Rotate(limit=90, border_mode=0, mask_value=0, p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.02,
                contrast_limit=0.02,
                brightness_by_max=True,
            ),
            A.RandomCrop(p=0.5, height=200, width=200),
        ]
    )

    transforms_aug = augmentations(image=img_t, mask=mask_t)
    img_t, mask_t = transforms_aug["image"], transforms_aug["mask"]

    return img_t, mask_t
