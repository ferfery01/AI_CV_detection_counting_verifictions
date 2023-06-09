from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

HUB_MODEL_ID = "facebook/sam-vit-huge"

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
MODEL_MPS = SamModel.from_pretrained(HUB_MODEL_ID).to(device)
MODEL_CPU = SamModel.from_pretrained(HUB_MODEL_ID).to("cpu")
PROCESSOR = SamProcessor.from_pretrained(HUB_MODEL_ID)


def get_image_embeddings(images: List[Image.Image]) -> torch.Tensor:
    """Get image embeddings for multiple images.

    Args:
        images: A list of PIL images. All the images must be of the same size.

    Returns:
        A tensor of shape (num_images, 256, 64, 64)
    """
    inputs = PROCESSOR(images, return_tensors="pt")
    image_embeddings = MODEL_MPS.get_image_embeddings(inputs["pixel_values"].to(device))

    return image_embeddings.cpu().detach()


def predict_masks(images: Union[np.ndarray, List[np.ndarray]]) -> List[torch.Tensor]:
    """Predict segmentation masks for each image. Return a list of masks.

    Args:
        images: A list of PIL images. All the images must be of the same size.

    Returns:
        A list of masks. Each mask is a tensor of shape (num_masks, height, width).
        The predicted masks are sorted in their IoU score order.
    """
    if isinstance(images, np.ndarray):
        images = [images]

    pill_width, pill_height, _ = images[0].shape
    center: Tuple[int, int] = pill_width // 2, pill_height // 2
    num_images: int = len(images)

    # The input points need to be in the format:
    #   nb_images, nb_predictions, nb_points_per_mask, 2
    input_points: List[List[List[int]]] = [[list(center)]] * num_images

    # Pre-process the images and the input points
    inputs = PROCESSOR(images, input_points=input_points, return_tensors="pt")
    image_embeddings = get_image_embeddings(images)

    # pop the pixel_values as they are not neded
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    # run the model in inference mode
    with torch.inference_mode():
        outputs = MODEL_CPU(**inputs)

    # post-process the masks to get the predicted masks
    masks: List[torch.Tensor] = PROCESSOR.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )

    return masks


def get_best_mask_per_images(masks: List[torch.Tensor]) -> List[np.ndarray]:
    """Get the mask with the maximum number of True values"""
    max_masks: List[np.ndarray] = []
    for mask in masks:
        # sum along second and third dimensions
        sums = torch.sum(mask[0], dim=(1, 2))

        # find the index of the maximum sum
        max_index = torch.argmax(sums)

        # get the mask with the maximum number of True values
        max_mask = mask[0][max_index, :, :]

        max_masks.append(max_mask.numpy().astype("uint8") * 255)

    return max_masks
