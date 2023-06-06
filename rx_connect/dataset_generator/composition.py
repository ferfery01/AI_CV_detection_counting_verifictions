import random
from typing import List, Sequence, Tuple

import numpy as np

from rx_connect.dataset_generator.object_overlay import add_pill_on_bg, verify_overlap
from rx_connect.dataset_generator.transform import resize_and_transform_pill

__all__: Sequence[str] = ("is_pill_within_image", "random_partition", "generate_image")


def is_pill_within_image(image: np.ndarray, mask: np.ndarray, position: Tuple[int, int]) -> bool:
    """
    Verifies whether an object along with its mask can fit inside a rectangular image.

    Args:
        image (numpy.ndarray): The rectangular image to be checked.
        mask (numpy.ndarray): The binary mask of the object.
        position (tuple): The (x,y) position of the object within the image.

    Returns:
        bool: True if the object along with its mask can fit inside the image, False otherwise.
    """
    # Get the minimum and maximum x and y coordinates of the mask
    y_min, y_max = np.min(np.where(mask > 0)[0]), np.max(np.where(mask > 0)[0])
    x_min, x_max = np.min(np.where(mask > 0)[1]), np.max(np.where(mask > 0)[1])
    obj_width, obj_height = x_max - x_min, y_max - y_min

    # Determine the dimensions of the mask and the rectangular image
    image_height, image_width = image.shape[:2]

    # Determine the bounding box of the object
    x, y = position
    xmin, ymin = x, y
    xmax, ymax = x + obj_width, y + obj_height

    # Check if the bounding box of the object is fully contained within the rectangular image
    if (xmin > 0 and ymin > 0) and (xmax < image_width and ymax < image_height):
        return True
    else:
        return False


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
    enable_edge_pills: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
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
        **kwargs: Keyword arguments to be passed to resize_and_transform_pill.

    Returns:
        A tuple containing the composed background image, the composition mask, the pill labels,
        and the pill areas.
    """
    h_bg, w_bg = bg_image.shape[0], bg_image.shape[1]

    pill_areas: List[int] = []
    pill_labels: List[int] = []
    count: int = 1

    for idx in range(n_pills):
        success: bool = False

        # Resize and transform the pill image and mask.
        pill_img_t, mask_t = resize_and_transform_pill(pill_image, pill_mask, **kwargs)

        for _ in range(max_attempts):
            # Randomly sample a position for the pill.
            # The position is sampled from a normal distribution with mean at the center of the background image
            # and standard deviation of a quarter of the background image's width and height.
            x, y = np.random.normal(loc=(w_bg / 2, h_bg / 2), scale=(w_bg / 4, h_bg / 4), size=(2,))
            x, y = np.clip(x, 0, w_bg), np.clip(y, 0, h_bg)

            # Check if the pill can fit inside the background image.
            if not enable_edge_pills and not is_pill_within_image(bg_image, mask_t, (x, y)):
                continue

            # Add the pill to the background image.
            bg_img_prev, comp_mask_prev = bg_image.copy(), comp_mask.copy()
            bg_image, comp_mask, added_mask = add_pill_on_bg(
                bg_image, comp_mask, pill_img_t, mask_t, int(x), int(y), count
            )

            # Verify that the pill does not overlap with other pills too much.
            if added_mask is not None and verify_overlap(comp_mask, pill_areas, max_overlap):
                pill_areas.append(np.count_nonzero(added_mask))
                if mode == "detection":  # all pills labeled as 1
                    pill_labels.append(1)
                elif mode == "segmentation":  # each pill label as it's own indexß
                    pill_labels.append(idx)
                else:
                    raise ValueError(f"Invalid mode {mode}. Must be either 'detection' or 'segmentation'.")
                success = True
                count += 1
                break
            else:
                bg_image, comp_mask = bg_img_prev.copy(), comp_mask_prev.copy()

        if not success:
            break

    return bg_image, comp_mask, pill_labels, pill_areas


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
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
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
        pill_areas: List of areas of the pills.
        pill_labels: List of labels of the pills.
    """

    bg_image = bg_image.copy()
    comp_mask = np.zeros(bg_image.shape[:2], dtype=np.uint8)

    # Randomly sample the number of pills to compose.
    num_pills = np.random.randint(min_pills, max_pills + 1)

    # Randomly sample the number of pills per type.
    pills_per_type = random_partition(num_pills, len(pill_images))

    pill_labels: List[int] = []
    pill_areas: List[int] = []
    count: int = 1

    for idx, n_pills in enumerate(pills_per_type):
        # Compose the pill on the background image.
        bg_image, comp_mask, labels, areas = _compose_pill_on_bg(
            bg_image,
            comp_mask,
            pill_images[idx],
            pill_masks[idx],
            mode,
            n_pills,
            max_overlap=max_overlap,
            max_attempts=max_attempts,
            enable_edge_pills=enable_edge_pills,
            **kwargs,
        )
        count += len(labels)
        pill_labels += labels
        pill_areas += areas

    return bg_image, comp_mask, pill_labels, pill_areas
