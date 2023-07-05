from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
from skimage.io import imread

# the output mask has for possible output values, marking each pixel
# in the mask as (1) definite background, (2) definite foreground,
# (3) probable background, and (4) probable foreground
GRABCUT_TYPES: Tuple[Tuple[str, int], ...] = (
    ("Definite Background", cv2.GC_BGD),
    ("Probable Background", cv2.GC_PR_BGD),
    ("Definite Foreground", cv2.GC_FGD),
    ("Probable Foreground", cv2.GC_PR_FGD),
)


def create_initial_mask_for_grabcut(image_path: Union[str, Path], threshold: float = 0.1) -> np.ndarray:
    """Creates an initial mask for the GrabCut algorithm.

    This function generates an initial binary mask for the GrabCut algorithm. Pixels in the
    grayscale image below the specified threshold are set as the background (0), while those
    above the threshold are set as the probable foreground (255).

    Args:
        image_path (Union[str, Path]): The path to the source image.
        threshold (float, optional): Intensity threshold for creating the initial binary mask.
            Pixels with a grayscale value below the threshold will be set as the background (0) and
            those above as the probable foreground (255). Defaults to 0.1.

    Returns:
        np.ndarray: The initialized binary mask for the GrabCut algorithm.
    """
    # Read the image in grayscale
    grayscale_image = imread(image_path, as_gray=True)

    # Initialize a new mask of the same size as the grayscale image with all zeros
    initial_mask = np.zeros_like(grayscale_image, dtype=np.uint8)

    # Set pixels with intensity less than the threshold as the background (0)
    initial_mask[grayscale_image < threshold] = 0

    # Set pixels with intensity greater than the threshold as probable foreground (255)
    initial_mask[grayscale_image > threshold] = 255

    return initial_mask


def apply_grabcut(image: np.ndarray, mask: np.ndarray, n_iter: int = 10) -> np.ndarray:
    """Apply GrabCut to the image."""
    # Set any value greater than zero to be foreground
    mask[mask > 0] = cv2.GC_PR_FGD

    # Set any value equal to zero to be background
    mask[mask == 0] = cv2.GC_BGD

    # allocate memory for two arrays that the GrabCut algorithm internally
    # uses when segmenting the foreground from the background
    bg_model = np.zeros((1, 65), dtype="float")
    fg_model = np.zeros((1, 65), dtype="float")

    # apply GrabCut using the the mask segmentation method
    (mask, bg_model, fg_model) = cv2.grabCut(
        image, mask, None, bg_model, fg_model, iterCount=n_iter, mode=cv2.GC_INIT_WITH_MASK
    )

    return mask


def post_process_mask(mask: np.ndarray) -> np.ndarray:
    """Remove the background from the mask."""
    # set all definite background and probable background pixels to 0
    # while definite foreground and probable foreground pixels are set
    # to 1, then scale the mask from the range [0, 1] to [0, 255]
    outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
    outputMask = (outputMask * 255).astype("uint8")

    return outputMask
