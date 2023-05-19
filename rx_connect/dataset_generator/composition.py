import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

from rx_connect.dataset_generator.io_utils import get_img_and_mask
from rx_connect.dataset_generator.object_overlay import add_pill_on_bg, verify_overlap
from rx_connect.dataset_generator.transform import resize_and_transform_pill


def is_object_mask_within_image(image: np.ndarray, mask: np.ndarray, position: Tuple[int, int]) -> bool:
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


def create_pill_comp(
    bg_img: np.ndarray,
    pill_mask_paths: List[Tuple[Path, Path]],
    n_pill_types: int,
    min_pills: int = 5,
    max_pills: int = 15,
    max_overlap: float = 0.2,
    max_attempts: int = 10,
    allow_pill_on_border: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """Create a composition of pills on a background image.

    Args:
        bg_img: The background image.
        pill_mask_paths: List of tuples of pill image and mask paths.
        n_pill_types: The number of different types of pills.
        min_pills: The minimum number of pills to compose.
        max_pills: The maximum number of pills to compose.
        max_overlap: The maximum allowed overlap between pills.
        max_attempts: The maximum number of attempts to compose a pill.
        allow_pill_on_border: whether to allow the pill object to be on the border of
            the background image.
        **kwargs: Keyword arguments for resize_and_transform_pill.

    Returns:
        bg_img: The background image with pills.
        composition_mask: The mask of the composition.
        pill_areas: List of areas of the pills.
        pill_labels: List of labels of the pills.
    """

    bg_img = bg_img.copy()
    h_bg, w_bg = bg_img.shape[0], bg_img.shape[1]
    comp_mask = np.zeros((h_bg, w_bg), dtype=np.uint8)

    pill_areas: List[int] = []
    pill_labels: List[int] = []

    # Randomly sample the number of pills to compose.
    num_pills = np.random.randint(min_pills, max_pills + 1)

    # Randomly sample the number of pills per type.
    pills_per_type = random_partition(num_pills, n_pill_types)

    count: int = 1
    for n_pills in pills_per_type:
        # Randomly sample a pill image and mask.
        idx = np.random.randint(len(pill_mask_paths))
        pill_img, mask = get_img_and_mask(pill_mask_paths[idx], thresh=0)

        for _ in range(1, n_pills + 1):
            success: bool = False
            for _ in range(max_attempts):
                # Randomly sample a position for the pill.
                # The position is sampled from a normal distribution with mean at the center of the background image
                # and standard deviation of a quarter of the background image's width and height.
                x, y = np.random.normal(loc=(w_bg / 2, h_bg / 2), scale=(w_bg / 4, h_bg / 4), size=(2,))
                x, y = np.clip(x, 0, w_bg), np.clip(y, 0, h_bg)

                # Resize and transform the pill image and mask.
                pill_img_t, mask_t = resize_and_transform_pill(pill_img, mask, **kwargs)

                # Check if the pill can fit inside the background image.
                if not allow_pill_on_border and not is_object_mask_within_image(bg_img, mask_t, (x, y)):
                    continue

                # Add the pill to the background image.
                bg_img_prev, comp_mask_prev = bg_img.copy(), comp_mask.copy()
                bg_img, comp_mask, added_mask = add_pill_on_bg(
                    bg_img, comp_mask, pill_img_t, mask_t, int(x), int(y), count
                )

                # Verify that the pill does not overlap with other pills too much.
                if added_mask is not None and verify_overlap(comp_mask, pill_areas, max_overlap):
                    pill_areas.append(np.count_nonzero(added_mask))
                    pill_labels.append(1)
                    success = True
                    count += 1
                    break
                else:
                    bg_img, comp_mask = bg_img_prev.copy(), comp_mask_prev.copy()

            if not success:
                break

    return bg_img, comp_mask, pill_labels, pill_areas
