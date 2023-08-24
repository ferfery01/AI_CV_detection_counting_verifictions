from typing import TypeVar

import cv2
import numpy as np
import torch

T = TypeVar("T", np.ndarray, torch.Tensor)


def to_three_channels(mask: T) -> T:
    """Converts a single channel mask to a three-channel mask. The resulting mask will have the
    same value for all three channels. This is useful for visualizing the mask on the image.

    Args:
        mask (T): The single channel mask to convert.

    Returns:
        T: The three-channel mask.

    Raises:
        TypeError: If the type of the mask is not supported.
        AssertionError: If the mask is not a single channel mask.
    """
    assert mask.ndim == 2, "Must be a single channel mask"
    if isinstance(mask, torch.Tensor):
        return mask.repeat(3, 1, 1)
    elif isinstance(mask, np.ndarray):
        return np.repeat(mask[..., None], 3, axis=2)
    else:
        raise TypeError("Unsupported type for mask")


def generate_grayscale_mask(image: np.ndarray, thresh: int = 0) -> np.ndarray:
    """Generates a boolean mask for the image."""
    # Convert the RGB image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a boolean mask where the grayscale image is greater than threshold
    boolean_mask = gray_image > thresh

    return boolean_mask


def fill_largest_contour(mask: np.ndarray, fill_value: int = 1) -> np.ndarray:
    """Fills the largest contour in the mask with the specified value and set
    anything outside the contour to 0.
    """
    mask = mask.astype(np.uint8).copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask)

    # Check if there are any contours
    if contours:
        # Sort the contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Draw the contours with a specific value
        cv2.drawContours(mask, contours, 0, fill_value, thickness=cv2.FILLED)

    return mask > 0
