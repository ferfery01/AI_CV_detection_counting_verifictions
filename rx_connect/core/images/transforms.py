import numpy as np
import torch
import torchvision.transforms.functional as TF


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
