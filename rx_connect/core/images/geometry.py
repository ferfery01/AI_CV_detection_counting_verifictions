from typing import List, Tuple, TypeVar, Union

import numpy as np
import torch

T = TypeVar("T", np.ndarray, torch.Tensor)


def expand_bounding_box(
    bbox: Union[List[float], torch.Tensor], image_shape: Tuple[int, int], expand_pixels: int
) -> List[int]:
    """Expands the bounding box by the specified number of pixels. If the bounding box is
    outside the image, it will be clipped to the image boundary.

    Args:
        bbox: The bounding box to expand.
        image_shape: The shape of the image.
        expand_pixels: The number of pixels to expand the bounding box by.

    Returns:
        The expanded bounding box.
    """
    height, width = image_shape
    return [
        max(0, int(bbox[0]) - expand_pixels),  # x_min
        max(0, int(bbox[1]) - expand_pixels),  # y_min
        min(width, int(bbox[2]) + expand_pixels),  # x_max
        min(height, int(bbox[3]) + expand_pixels),  # y_max
    ]


def extract_ROI(image: T, bbox: List[int]) -> T:
    """Extracts the ROI from the image defined by the bounding box."""
    x_min, y_min, x_max, y_max = bbox
    if isinstance(image, torch.Tensor):
        return image[:, y_min:y_max, x_min:x_max]
    elif isinstance(image, np.ndarray):
        return image[y_min:y_max, x_min:x_max]
    else:
        raise TypeError(f"image must be either a torch.Tensor or a np.ndarray, but got {type(image)}")
