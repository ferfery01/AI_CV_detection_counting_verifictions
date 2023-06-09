import random
from typing import List, NamedTuple, Sequence

import numpy as np

from rx_connect.dataset_generator.object_overlay import (
    check_overlap,
    is_pill_within_background,
    overlay_image_onto_background,
)
from rx_connect.dataset_generator.transform import resize_and_transform_pill

__all__: Sequence[str] = ("random_partition", "generate_image", "ImageComposition")


class ImageComposition(NamedTuple):
    """The image, mask, and label IDs of the composed image."""

    image: np.ndarray
    """Image containing the composed pills.
    """
    mask: np.ndarray
    """Mask containing the composed pills.
    """
    labels: List[int]
    """List of label IDs of the composed pills.
    """


def random_partition(number: int, num_parts: int) -> List[int]:
    """Generates a list of random integers that add up to a specified number.

    Args:
        number (int): The number to be divided into multiple parts.
        num_parts (int): The number of parts to divide the number into.

    Returns:
        A list of num_parts random integers that add up to number.
    """

    parts: List[int] = [0] * num_parts
    for i in range(num_parts - 1):
        parts[i] = random.randint(0, number)
        number -= parts[i]
    parts[num_parts - 1] = number

    return parts


def _compose_pill_on_bg(
    bg_image: np.ndarray,
    comp_mask: np.ndarray,
    pill_image: np.ndarray,
    pill_mask: np.ndarray,
    mode: str,
    n_pills: int,
    max_overlap: float = 0.2,
    max_attempts: int = 10,
    start_index: int = 1,
    enable_edge_pills: bool = False,
    **kwargs,
) -> ImageComposition:
    """Compose n_pills pills on a background image.

    Args:
        bg_image: The background image.
        comp_mask: The composition mask.
        pill_image: The pill image.
        pill_mask: The pill mask.
        n_pills: The number of pills to compose.
        mode: The composition mode. Can be either "detection" or "segmentation".
        max_overlap: The maximum allowed overlap between pills.
        max_attempts: The maximum number of attempts to compose a pill.
        enable_edge_pills: Whether to allow pills to be placed on the border of the background image.
        start_index: The starting index for the pill labels.
        **kwargs: Keyword arguments to be passed to resize_and_transform_pill.

    Returns:
        The composed image, mask, and label IDs.
    """
    h_bg, w_bg = bg_image.shape[:2]
    h_pill, w_pill = pill_image.shape[:2]

    label_ids: List[int] = []
    count: int = 1

    for _ in range(n_pills):
        # Resize and transform the pill image and mask.
        pill_img_t, pill_mask_t = resize_and_transform_pill(pill_image, pill_mask, **kwargs)

        # Attempt to compose the pill on the background image.
        for _ in range(max_attempts):
            # Randomly sample a position for the pill.
            # The position is sampled from a normal distribution with mean at the center of the background image
            # and standard deviation of a quarter of the background image's width and height.
            x, y = np.random.normal(loc=(w_bg / 2, h_bg / 2), scale=(w_bg / 4, h_bg / 4), size=(2,)).astype(
                int
            )
            top_left = np.clip(x, -w_pill, w_bg), np.clip(y, -h_pill, h_bg)

            # Check if the pill can fit inside the background image.
            if not is_pill_within_background(bg_image, pill_mask_t, top_left, enable_edge_pills):
                continue

            # Verify that the new pill does not overlap with the existing pills.
            if not check_overlap(pill_mask_t, comp_mask, top_left, max_overlap):
                continue

            # Add the pill to the background image.
            bg_image, comp_mask = overlay_image_onto_background(
                bg_image, comp_mask, pill_img_t, pill_mask_t, top_left, start_index + count
            )

            # Update the label ids depending on the composition mode.
            if mode == "detection":  # all pills labeled as 1
                label_ids.append(1)
            elif mode == "segmentation":  # each pill label as it's own index
                label_ids.append(start_index + count)
            else:
                raise ValueError(f"Invalid mode {mode}. Must be either 'detection' or 'segmentation'.")
            count += 1
            break

    return ImageComposition(bg_image, comp_mask, label_ids)


def generate_image(
    bg_image: np.ndarray,
    pill_images: List[np.ndarray],
    pill_masks: List[np.ndarray],
    mode: str = "detection",
    min_pills: int = 5,
    max_pills: int = 15,
    max_overlap: float = 0.2,
    max_attempts: int = 10,
    enable_edge_pills: bool = False,
    **kwargs,
) -> ImageComposition:
    """Create a composition of pills on a background image.

    Args:
        bg_image: The background image.
        pill_images: A list of pill images.
        pill_masks: A list of pill masks.
        mode: Detection data generation or segmentation data generation mode flag.
        min_pills: The minimum number of pills to compose.
        max_pills: The maximum number of pills to compose.
        max_overlap: The maximum allowed overlap between pills.
        max_attempts: The maximum number of attempts to compose a pill.
        enable_edge_pills: whether to allow the pill object to be on the border of
            the background image.
        **kwargs: Keyword arguments for resize_and_transform_pill.

    Returns:
        bg_image: The background image with pills.
        composition_mask: The mask of the composition.
            - If it is detection mode, all pills will be labeled as 1;
            - It it is segmentation mode, pills will be labeled as it's index, from 1, 2, 3..., n_pills.
        pill_labels: List of labels of the pills.
    """

    bg_image = bg_image.copy()
    comp_mask = np.zeros(bg_image.shape[:2], dtype=np.uint8)

    # Randomly sample the number of pills to compose.
    num_pills = np.random.randint(min_pills, max_pills + 1)

    # Randomly sample the number of pills per type.
    pills_per_type = random_partition(num_pills, len(pill_images))

    label_ids: List[int] = []

    for idx, n_pills in enumerate(pills_per_type):
        # Compose the pill on the background image.
        bg_image, comp_mask, labels = _compose_pill_on_bg(
            bg_image,
            comp_mask,
            pill_images[idx],
            pill_masks[idx],
            mode,
            n_pills,
            max_overlap=max_overlap,
            max_attempts=max_attempts,
            enable_edge_pills=enable_edge_pills,
            start_index=len(label_ids),
            **kwargs,
        )
        label_ids += labels

    return ImageComposition(bg_image, comp_mask, label_ids)
