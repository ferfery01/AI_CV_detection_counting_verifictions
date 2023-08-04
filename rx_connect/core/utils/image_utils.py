import io
from pathlib import Path
from typing import List, Union
from urllib.parse import urlparse

import numpy as np
import requests
import torch
import torchshow as ts
from PIL import Image
from tqdm import tqdm

from rx_connect.tools.logging import setup_logger

logger = setup_logger()

IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".cr2"]
"""The list of image file extensions.
"""


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


def download_image(url: str, path: Union[str, Path], verbose: bool = False) -> None:
    """Download an image from a URL and save it to a file.

    Args:
        url (str): The URL of the image.
        path (Union[str, Path]): The path to save the image to.
        verbose (bool, optional): Whether to print the download progress.

    Raises:
        requests.HTTPError: If the HTTP request fails.
    """
    path = Path(path)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx

        file_size = int(response.headers.get("Content-Length", 0))

        with tqdm(total=file_size, unit="iB", unit_scale=True, disable=not verbose) as progress:
            # Open a binary file in write mode
            with path.open("wb") as file:
                # Write data read from the response
                for data in response.iter_content(1024):
                    size = file.write(data)
                    progress.update(size)
        if verbose:
            logger.info(f"Downloaded image from {url} to {path}")
    except requests.HTTPError as http_err:
        logger.exception(f"HTTP error occurred: {http_err}", exc_info=True)
    except Exception as err:
        logger.exception(f"An error occurred: {err}", exc_info=True)


def is_url(url: str) -> bool:
    """Check if the given string is a URL.
    :param url:  String to check.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc, result.path])
    except Exception:
        return False


def load_pil_image_from_str(image_str: str) -> Image.Image:
    """Load an image based on a string (local file path or URL)."""

    if is_url(image_str):
        response = requests.get(image_str, stream=True)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    else:
        return Image.open(image_str)


def display_image(image: Union[str, Path, np.ndarray, Image.Image, torch.Tensor]) -> None:
    """Display an image in a Jupyter notebook.

    Supported image types include:
        - numpy.ndarray:    A numpy array representing the image
        - torch.Tensor:     A PyTorch tensor representing the image
        - PIL.Image.Image:  A PIL Image object
        - str:              A string representing either a local file path or a URL to an image
        - Path:             A Path object representing a local file path
    """
    if isinstance(image, torch.Tensor):
        image = Image.fromarray(image.numpy())
    elif isinstance(image, Path):
        image = Image.open(image)
    elif isinstance(image, str):
        image = load_pil_image_from_str(image_str=image)
    ts.show(image)
