from typing import List

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from segment_anything import SamAutomaticMaskGenerator, build_sam

from rx_connect.core.types.segment import SamHqSegmentResult
from rx_connect.tools.logging import setup_logger

logger = setup_logger()

# Download the SAM-HQ model from HuggingFace Hub and load it into memory
_ckpt_path = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_h_4b8939.pth")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using {device} for SAM-HQ model inference.")
model = build_sam(checkpoint=_ckpt_path).to(device)
MASK_GENERATOR = SamAutomaticMaskGenerator(model, min_mask_region_area=5000)


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


def get_mask_from_SAM(image: np.ndarray) -> np.ndarray:
    """Get the mask from the SAM model. The function uses a SAM-HQ model to generate
    multiple masks and then selects the best one.

    Args:
        image (np.ndarray): The input image from which to generate masks.

    Returns:
        np.ndarray: The best mask (inverted) generated from the SAM model, represented
            as a 2D numpy array.
    """
    # Generate masks from the input image using the SAM model
    masks = MASK_GENERATOR.generate(image)

    # Get the best mask from the list of masks generated
    return get_best_mask(masks)
