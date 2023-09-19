from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchshow as ts
from PIL import Image
from skimage.io import imread

from rx_connect.core.images.io import load_pil_image_from_str

plt.rcParams["savefig.bbox"] = "tight"
"""Set the bounding box for the saved figure to tightly fit the figure
"""


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


def visualize_masks(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    anno_file: Union[str, Path],
    figsize: Tuple[int, int] = (15, 10),
    cmap: Optional[str] = None,
    wspce: float = 0.01,
) -> None:
    """Visualize an image along with its corresponding mask and annotated polygons.

    Args:
        image_path (Union[str, Path]): Path to the image file.
        mask_path (Union[str, Path]): Path to the mask file (numpy array).
        anno_file (Union[str, Path]): Path to the annotation file containing polygon coordinates.
        figsize (Tuple[int, int]): The figure size (width, height) in inches.
        cmap (str, optional): The colormap to use for the mask plot. Defaults to None. Can be one of
            the following: https://matplotlib.org/stable/tutorials/colors/colormaps.html
        wspce (float): The width of the space between the two subplots.

    Displays:
        A matplotlib plot with two subplots: the original image and the mask with overlaid polygons.
    """
    image, mask = imread(image_path), imread(mask_path)
    height, width = mask.shape

    # Load annotations and split into lines
    annotations = Path(anno_file).read_text().splitlines()

    # Create a figure with two subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Reduce the gap between the two columns by setting wspace
    plt.subplots_adjust(wspace=wspce)

    # Display the image and mask
    ax1.imshow(image)
    ax1.axis("off")
    ax2.imshow(mask, cmap=cmap)

    # Iterate over the annotations and plot each polygon
    for annotation in annotations:
        # Split the annotation string and convert to float
        coords = [float(x) for x in annotation.split()[1:]]

        # Rescale the coordinates to actual pixel values
        coords[::2] = [x * width for x in coords[::2]]
        coords[1::2] = [y * height for y in coords[1::2]]

        # Create a polygon patch and add it to the plot
        poly = patches.Polygon(
            [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)],
            closed=True,
            edgecolor="r",
            facecolor="none",
        )
        ax2.add_patch(poly)

    # Set plot properties
    ax2.set_xlim(0, width)
    ax2.set_ylim(0, height)
    ax2.invert_yaxis()
    ax2.axis("off")

    # Display the plot
    plt.show()


def visualize_gallery(show_imgs: List[np.ndarray], img_per_row: int = 5) -> None:
    """Visualizes a series of image, print them 5 (or as specified) per row.

    Args:
        show_imgs (List[np.ndarray]): the list of images to visualize.
        img_per_row (5): the number of images to visualize in each row.
    """
    show_img_reordered = [
        show_imgs[i * img_per_row : i * img_per_row + img_per_row]
        for i in range(-(len(show_imgs) // -img_per_row))
    ]
    ts.show(show_img_reordered)
