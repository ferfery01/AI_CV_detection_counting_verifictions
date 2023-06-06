import numpy as np
from scipy.ndimage import rotate

"""This module contains utility functions for image transforms."""


def rotate_image(image: np.ndarray, angle: int = 0) -> np.ndarray:
    """Rotate an image by a certain angle and return the result as an np.uint8 array.

    Args:
        image: The image to rotate. Must be an np.uint8 array.
        angle: The angle to rotate the image, in degrees. Positive values rotate counter-
            clockwise, and negative values rotate clockwise.

    Returns:
        The rotated image as an np.uint8 array.
    """

    # Rotate the image
    rotated_image = rotate(image, angle, reshape=False)

    # Ensuring that all values are within the correct range
    rotated_image = np.clip(rotated_image, 0, 255)

    return rotated_image.astype(np.uint8)  # Convert to uint8
