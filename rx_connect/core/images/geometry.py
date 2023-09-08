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


def center_distance(bbox: Tuple[int, int, int, int], image_center: Tuple[float, float]) -> float:
    """Computes the Euclidean distance between the center of a bounding box and the center of the image.

    Args:
        bbox (tuple): Bounding box in the format (min_row, min_col, max_row, max_col).
        image_center (tuple): Coordinates of the image center (center_y, center_x).

    Returns:
        float: Euclidean distance between the centers.
    """
    center_y, center_x = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return np.sqrt((center_x - image_center[1]) ** 2 + (center_y - image_center[0]) ** 2)
