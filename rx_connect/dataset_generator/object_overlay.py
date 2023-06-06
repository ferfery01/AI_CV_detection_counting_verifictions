from typing import List, Optional, Tuple

import numpy as np


def add_pill_on_bg(
    img_bg: np.ndarray,
    mask_comp: np.ndarray,
    img_pill: np.ndarray,
    mask_pill: np.ndarray,
    x: int,
    y: int,
    idx: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Add the pill object to the background image.

    Args:
        img_bg (np.ndarray): background image.
        mask_comp (np.ndarray): mask of the background image.
        img_pill (np.ndarray): pill image.
        mask_pill (np.ndarray): binary mask of the pill
        x (int): x coordinate of the center of the pill object.
        y (int): y coordinate of the center of the pill object.
        idx (int): index of the pill object.

    Returns:
        - the background image composition (background + pills),
        - the mask of the background image composition, and the mask of the pill image, and
        - the mask of the last pill image
    """
    h_bg, w_bg = img_bg.shape[:2]
    h_pill, w_pill = img_pill.shape[:2]
    mask_added: Optional[np.ndarray] = None

    # Convert to RGB mask
    mask_b = mask_pill == 1
    mask_rgb_b = np.stack([mask_b, mask_b, mask_b], axis=2)

    if (x >= 0 and y >= 0) and (x < w_bg and y < h_bg):
        # part of the image which gets into the frame of img_bg along y-axis
        h_part = h_pill - max(0, y + h_pill - h_bg)
        # part of the image which gets into the frame of img_bg along x-axis
        w_part = w_pill - max(0, x + w_pill - w_bg)

        # Add the pill object to the background image and compose the mask
        img_bg[y : y + h_part, x : x + w_part, :] = (
            img_bg[y : y + h_part, x : x + w_part, :] * (~mask_rgb_b[:h_part, :w_part, :])
            + (img_pill * mask_rgb_b)[:h_part, :w_part, :]
        )
        mask_comp[y : y + h_part, x : x + w_part] = (
            mask_comp[y : y + h_part, x : x + w_part] * (~mask_b[:h_part, :w_part])
            + (idx * mask_b)[:h_part, :w_part]
        )
        mask_added = mask_pill[:h_part, :w_part]

    elif (x < 0 and y < 0) and (x + w_pill > 0 and y + h_pill > 0):
        h_part = h_pill + y
        w_part = w_pill + x

        img_bg[:h_part, :w_part, :] = (
            img_bg[:h_part, :w_part, :] * (~mask_rgb_b[h_pill - h_part : h_pill, w_pill - w_part : w_pill, :])
            + (img_pill * mask_rgb_b)[h_pill - h_part : h_pill, w_pill - w_part : w_pill, :]
        )
        mask_comp[:h_part, :w_part] = (
            mask_comp[:h_part, :w_part] * (~mask_b[h_pill - h_part : h_pill, w_pill - w_part : w_pill])
            + (idx * mask_b)[h_pill - h_part : h_pill, w_pill - w_part : w_pill]
        )
        mask_added = mask_pill[h_pill - h_part : h_pill, w_pill - w_part : w_pill]

    elif (x < 0 and y >= 0) and (x + w_pill > 0 and y < h_bg):
        h_part = h_pill - max(0, y + h_pill - h_bg)
        w_part = w_pill + x

        img_bg[y : y + h_part, :w_part, :] = (
            img_bg[y : y + h_part, :w_part, :] * (~mask_rgb_b[:h_part, w_pill - w_part : w_pill, :])
            + (img_pill * mask_rgb_b)[:h_part, w_pill - w_part : w_pill, :]
        )
        mask_comp[y : y + h_part, :w_part] = (
            mask_comp[y : y + h_part, :w_part] * (~mask_b[:h_part, w_pill - w_part : w_pill])
            + (idx * mask_b)[:h_part, w_pill - w_part : w_pill]
        )
        mask_added = mask_pill[:h_part, w_pill - w_part : w_pill]

    elif (x >= 0 and y < 0) and (x < w_bg and y + h_pill > 0):
        h_part = h_pill + y
        w_part = w_pill - max(0, x + w_pill - w_bg)

        img_bg[:h_part, x : x + w_part, :] = (
            img_bg[:h_part, x : x + w_part, :] * (~mask_rgb_b[h_pill - h_part : h_pill, :w_part, :])
            + (img_pill * mask_rgb_b)[h_pill - h_part : h_pill, :w_part, :]
        )
        mask_comp[:h_part, x : x + w_part] = (
            mask_comp[:h_part, x : x + w_part] * (~mask_b[h_pill - h_part : h_pill, :w_part])
            + (idx * mask_b)[h_pill - h_part : h_pill, :w_part]
        )
        mask_added = mask_pill[h_pill - h_part : h_pill, :w_part]

    return img_bg, mask_comp, mask_added


def verify_overlap(mask_comp: np.ndarray, pill_areas: List[int], overlap_degree: float = 0.3) -> bool:
    """Check if any of the previous pills overlaps more than overlap_degree with the current pill.

    Args:
        mask_comp (np.ndarray): mask of the background image composition.
        pill_areas (List[int]): List of pill areas in order of thier addition to the background image.
        overlap_degree (float, optional): overlap degree threshold. Defaults to 0.3.

    Returns:
        bool: True if the current pill overlaps with any of the previous pills more than overlap_degree,
        False otherwise.
    """
    # If there are no previous pills, return True
    if len(pill_areas) == 0:
        return True

    pill_ids = np.unique(mask_comp).astype(np.uint8)[1:-1]
    masks = mask_comp == pill_ids[:, None, None]

    if len(np.unique(mask_comp)) != np.max(mask_comp) + 1:
        return False

    overlap: bool = True
    for idx, mask in enumerate(masks):
        if np.count_nonzero(mask) / pill_areas[idx] < 1 - overlap_degree:
            overlap = False
            break

    return overlap
