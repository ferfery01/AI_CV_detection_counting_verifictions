from typing import List

import numpy as np
from huggingface_hub import hf_hub_download
from segment_anything import SamAutomaticMaskGenerator, build_sam

from rx_connect.core.types.segment import SamHqSegmentResult
from rx_connect.core.utils.sam_utils import get_best_mask
from rx_connect.tools.device import get_best_available_device
from rx_connect.tools.logging import setup_logger

logger = setup_logger()

# Download the SAM-HQ model from HuggingFace Hub and load it into memory
_ckpt_path = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_h_4b8939.pth")
device = get_best_available_device()
logger.info(f"Using {device} for SAM-HQ model inference.")
model = build_sam(checkpoint=_ckpt_path).to(device)
MASK_GENERATOR = SamAutomaticMaskGenerator(model, min_mask_region_area=5000)


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
    masks: List[SamHqSegmentResult] = MASK_GENERATOR.generate(image)  # type: ignore

    # Get the best mask from the list of masks generated
    return get_best_mask(masks)
