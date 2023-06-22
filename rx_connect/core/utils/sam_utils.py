from typing import List

import numpy as np

from rx_connect.core.types.segment import SamHqSegmentResult


def get_best_mask(masks: List[SamHqSegmentResult]) -> np.ndarray:
    """Get the best mask from a list of masks. The best mask is the one with the highest
    sum of pixels.

    Args:
        masks (List[SamHqSegmentResult]): A list of SamHqSegmentResult. Each dictionary should have a
            key "segmentation" with a 2D numpy array as its value.

    Returns:
        np.ndarray: The best mask (inverted) from the list of masks, represented as a 2D
        numpy array.
    """
    # Stack all the segmentation masks into a single 3D array
    seg_masks = np.stack([mask["segmentation"] for mask in masks], axis=0)

    # Find the index of the mask with the highest sum of pixels.
    # This mask is considered as the best mask.
    best_mask_idx = np.argmax(seg_masks.sum(axis=(1, 2)))

    # Extract the best mask
    best_seg_mask = seg_masks[best_mask_idx]

    # Invert the mask and return it
    return np.logical_not(best_seg_mask)
