from typing import Optional, Sequence, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.ndimage import rotate


def pad_image(image: np.ndarray, pad_width: int, pad_value: int = 0) -> np.ndarray:
    """Pad an image with the specified value.

    Args:
        image (np.ndarray): The image to pad.
        pad_width (int): The number of pixels to pad the image on each side.
        pad_value (int, optional): The value to use for padding. Defaults to 0.

    Returns:
        np.ndarray: The padded image.
    """
    return np.pad(
        image,
        pad_width=((pad_width, pad_width), (pad_width, pad_width), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )


def resize_to_square(
    image: np.ndarray,
    mode: str = "constant",
    cval: Optional[Union[int, Sequence[int], Sequence[Sequence[int]]]] = 0,
) -> np.ndarray:
    """Pads the shorter dimension of a given image to make it square. The padding is applied to both sides
    of the shorter dimension, and the original aspect ratio of the image is preserved.

    Args:
        image (np.ndarray): The input image, which can have one (grayscale) or three (color) channels.
        mode (str, optional): The method used for padding. Can be one of the modes supported by `np.pad`.
            Defaults to 'constant'.
        cval (Optional[Union[int, Sequence[int], Sequence[Sequence[int]]]], optional): The constant values to
            use for padding when mode is 'constant'. Can be a scalar or sequence of length equal to the number
            of dimensions in the image. Defaults to 0.

    Returns:
        np.ndarray: The padded square image.
    """
    # Find the shape of the image
    height, width = image.shape[:2]

    # Determine the size of padding for the shorter dimension
    difference = abs(height - width)
    padding_size = difference // 2
    padding_extra = difference % 2

    # Define padding for height or width depending on which is smaller
    if height < width:
        padding = [(padding_size, padding_size + padding_extra), (0, 0)]
    else:
        padding = [(0, 0), (padding_size, padding_size + padding_extra)]

    # Add padding for the third dimension if the image is colored (3 channels)
    if image.ndim == 3:
        padding += [(0, 0)]

    # Pad the image using NumPy's pad function and the padding sizes determined earlier
    square_image = np.pad(image, padding, mode=mode, constant_values=cval)  # type: ignore

    return square_image


def resize_and_center(
    image: torch.Tensor,
    target_height: int,
    target_width: int,
    interpolation: TF.InterpolationMode = TF.InterpolationMode.BILINEAR,
    fill: int = 0,
) -> torch.Tensor:
    """Resize the given tensor to the target dimensions while keeping the central object
    centered.

    Args:
        image (torch.Tensor): The image tensor to resize.
        target_height (int): The target height.
        target_width (int): The target width.
        interpolation (TF.InterpolationMode, optional): The interpolation mode to use. Defaults to
            TF.InterpolationMode.BILINEAR.
        fill (int, optional): The value to use for padding. Defaults to 0.

    Returns:
        torch.Tensor: The resized image tensor.
    """
    # Get the original dimensions
    _, original_height, original_width = image.shape

    # Calculate the scaling factors for height and width
    scale_height = target_height / original_height
    scale_width = target_width / original_width

    # Use the smaller scaling factor to maintain aspect ratio
    scale = min(scale_height, scale_width)
    new_height, new_width = int(original_height * scale), int(original_width * scale)

    # Resize the image with the new dimensions
    resized_image = TF.resize(image, (new_height, new_width), interpolation=interpolation)

    # Calculate padding and cropping to center the image
    pad_top = max((target_height - new_height) // 2, 0)
    pad_bottom = target_height - new_height - pad_top
    pad_left = max((target_width - new_width) // 2, 0)
    pad_right = target_width - new_width - pad_left

    # Pad the resized image if needed
    padded_image = TF.pad(resized_image, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)

    return padded_image


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
