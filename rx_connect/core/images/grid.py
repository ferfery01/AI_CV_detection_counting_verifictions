from typing import List

import numpy as np

from rx_connect.core.images.transforms import pad_image
from rx_connect.tools.logging import setup_logger

logger = setup_logger()


def create_square_grid(
    images: List[np.ndarray], grid_shape: int, pad_width: int = 15, pad_value: int = 0
) -> np.ndarray:
    """Create a grid of images.

    The grid is created by stacking the images in the list horizontally and then
    vertically.  If the number of images is not equal to the product of the grid
    shape, the list of images is padded with empty images until the number of images
    is equal to the product of the grid shape.

    Args:
        images (List[np.ndarray]): The list of images to create a grid from.
        grid_shape (int): The shape of the grid (rows, columns).
        pad_width (int): The number of pixels to pad the images on each side. Defaults to 15.
        pad_value (int, optional): The value to use for padding. Defaults to 0.

    Returns:
        np.ndarray: The grid of images.
    """
    n_images: int = grid_shape**2
    images_list: List[np.ndarray] = [pad_image(image, pad_width, pad_value) for image in images]

    # If the number of images is not equal to the product of the grid shape, pad the list with
    # empty images until the number of images is equal to the product of the grid shape
    if len(images_list) != n_images:
        tmp_image = pad_value * np.ones_like(images_list[0], dtype=np.uint8)
        images_list += [tmp_image] * (n_images - len(images_list))

    # Create a list of lists for each row of images
    image_grid = [images_list[idx : idx + grid_shape] for idx in range(0, n_images, grid_shape)]

    # Combine images in a grid
    return np.vstack([np.hstack(row_images) for row_images in image_grid])


def unpack_images_from_grid(image: np.ndarray, grid_shape: int, pad_width: int = 15) -> List[np.ndarray]:
    """Unpack images from a grid. The images are cropped to remove the padding.

    Args:
        image (np.ndarray): The grid of images.
        grid_shape (int): The shape of the grid (rows, columns).
        pad_width (int): The number of pixels to pad the images on each side.

    Returns:
        List[np.ndarray]: The list of images.
    """
    # Split combined image into the original images
    split_images = [np.hsplit(row, grid_shape) for row in np.vsplit(image, grid_shape)]

    # Flatten the list of lists into a single list of images
    retrieved_images = [img for sublist in split_images for img in sublist]

    # Crop the padding from each image
    return [img[pad_width:-pad_width, pad_width:-pad_width] for img in retrieved_images]
