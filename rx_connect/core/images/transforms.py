import numpy as np


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
