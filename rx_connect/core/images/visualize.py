from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchshow as ts
from PIL import Image

from rx_connect.core.images.io import load_pil_image_from_str


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
