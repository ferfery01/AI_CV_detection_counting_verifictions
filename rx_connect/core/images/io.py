import io
import random
from pathlib import Path
from typing import List, Union

import numpy as np
import requests
import torch
from PIL import Image
from skimage.io import imread
from tqdm import tqdm

from rx_connect.core.utils import is_url
from rx_connect.tools.logging import setup_logger

logger = setup_logger()

IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".cr2"]
"""The list of image file extensions.
"""


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


def load_pil_image_from_str(image_str: str) -> Image.Image:
    """Load an image based on a string (local file path or URL)."""

    if is_url(image_str):
        response = requests.get(image_str, stream=True)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    else:
        return Image.open(image_str)


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Loads an image from the given path.

    For images with 4 channels, the alpha channel is removed.
    For images with multiple channels i.e. a video, a random frame is selected.
    """
    image = imread(image_path)

    if len(image.shape) == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]
    elif len(image.shape) == 4:
        idx = random.randint(0, image.shape[0])
        image = image[idx, :, :, :3]

    return image


def img_to_tensor(image: np.ndarray, dtype: type = np.float32) -> torch.Tensor:
    """Converts numpy image (RGB, BGR, Grayscale, Mask) to a `torch.Tensor`. The numpy `HWC`
    image is converted to `CHW` tensor. If the image is in `HW` format (grayscale, mask), it will
    be converted to pytorch `HW` tensor.

    Args:
        image (np.ndarray): The image to convert.
        dtype (np.dtype, optional): The dtype of the tensor. Defaults to np.float32.

    Returns:
        torch.Tensor of shape (C, H, W) or (H, W).
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must have shape (H, W) or (H, W, C), got {image.shape}")

    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image.astype(dtype, copy=False)).float()
