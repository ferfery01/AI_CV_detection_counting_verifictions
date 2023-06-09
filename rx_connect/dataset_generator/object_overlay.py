from typing import Tuple

import numpy as np


def is_pill_within_background(
    bg_image: np.ndarray,
    pill_mask: np.ndarray,
    top_left_position: Tuple[int, int],
    allow_edge_pills: bool = False,
) -> bool:
    """Checks if a pill specified by its mask fits within a background at the given position.

    Args:
        background (np.ndarray): The target background to place the pill onto.
        pill_mask (np.ndarray): The binary mask of the pill, where '1' represents the pill.
        top_left_position (Tuple[int, int]): The position of the top left corner of the pill
            in the background.
        allow_edge_pills (bool):
            - If True, allows pills that can partially be placed on the background.
            - If False, doesn't allow partial pills to be placed at the border of the background.

    Returns:
        bool: True if the pill can be fully or partially (depending on 'allow_edge_pills')
        contained within the background, False otherwise.
    """
    x, y = top_left_position

    # Fetch the indices where pill_mask has a non-zero value.
    mask_y_indices, mask_x_indices = np.where(pill_mask > 0)

    # Determine the dimensions of the pill using its mask.
    pill_height = np.max(mask_y_indices) - np.min(mask_y_indices)
    pill_width = np.max(mask_x_indices) - np.min(mask_x_indices)

    # Determine the dimensions of the target background.
    bg_heigth, bg_width = bg_image.shape[:2]

    # Determine the potential boundaries of the pill within the background.
    top_left_x = x + mask_x_indices.min()
    top_left_y = y + mask_y_indices.min()
    bottom_right_x = top_left_x + pill_width
    bottom_right_y = top_left_y + pill_height

    # Check if the pill fits within the background. The pill fits if its top left position
    # is within the background and its bottom right position does not exceed the background's dimensions.
    # The 'allow_edge_pills' flag determines if the pill is allowed to partially exist on the background's edge.
    if allow_edge_pills:
        # If we allow edge pills, then it's enough for the pill to have some intersection with the background.
        return (top_left_x < bg_width and top_left_y < bg_heigth) and (
            bottom_right_x > 0 and bottom_right_y > 0
        )
    else:
        # If we don't allow edge pills, the entire pill needs to be inside the background.
        return (
            top_left_x >= 0 and top_left_y >= 0 and bottom_right_x <= bg_width and bottom_right_y <= bg_heigth
        )


def overlay_image_onto_background(
    bg_image: np.ndarray,
    bg_mask: np.ndarray,
    pill_img: np.ndarray,
    pill_mask: np.ndarray,
    top_left_corner: Tuple[int, int],
    pill_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Overlay the given image onto the background image using the specified mask.

    Args:
        bg_image (np.ndarray): The original background image.
        bg_mask (np.ndarray): Mask for the background image.
        pill_img (np.ndarray): Image to overlay on the background.
        pill_mask (np.ndarray): Binary mask for the image to overlay.
        top_left_corner (Tuple[int, int]): Coordinates for the top left corner of the overlay.
        pill_id (int): Identifier for the overlay image.

    Returns:
        np.ndarray: The resulting image after overlaying the input image onto the background.
        np.ndarray: The updated mask after overlaying the input image onto the background.
    """
    x_coord, y_coord = top_left_corner
    bg_height, bg_width = bg_image.shape[:2]
    pill_height, pill_width = pill_img.shape[:2]

    # Converting binary mask to RGB mask for compatibility with RGB images
    mask_bool = pill_mask == 1
    mask_rgb = np.stack([mask_bool] * 3, axis=-1)

    # Compute the effective dimensions of the overlay within the bounds of the background
    eff_width = min([pill_width, pill_width + x_coord, bg_width - x_coord])
    eff_height = min([pill_height, pill_height + y_coord, bg_height - y_coord])

    # Compute the effective regions in both the background and overlay images
    bg_top_x, bg_top_y = max(0, x_coord), max(0, y_coord)
    pill_top_x, pill_top_y = max(0, -x_coord), max(0, -y_coord)

    # Apply the effective overlay mask to both images
    effective_mask = mask_bool[pill_top_y : pill_top_y + eff_height, pill_top_x : pill_top_x + eff_width]
    effective_rgb_mask = mask_rgb[pill_top_y : pill_top_y + eff_height, pill_top_x : pill_top_x + eff_width]

    # Resize pill image to the size of the effective mask
    resized_pill_img = pill_img[pill_top_y : pill_top_y + eff_height, pill_top_x : pill_top_x + eff_width]

    # Resize background image to the size of the effective overlay
    resized_bg_img = bg_image[bg_top_y : bg_top_y + eff_height, bg_top_x : bg_top_x + eff_width]

    # Compute the effective overlay, taking care to avoid altering the background where the mask is False
    bg_image[bg_top_y : bg_top_y + eff_height, bg_top_x : bg_top_x + eff_width] = (
        resized_bg_img * (~effective_rgb_mask) + resized_pill_img * effective_rgb_mask
    )

    # Update the mask to reflect the overlay, again only altering masked locations
    bg_mask[bg_top_y : bg_top_y + eff_height, bg_top_x : bg_top_x + eff_width] = (
        bg_mask[bg_top_y : bg_top_y + eff_height, bg_top_x : bg_top_x + eff_width] * (~effective_mask)
        + pill_id * effective_mask
    )

    return bg_image, bg_mask


def check_overlap(
    mask: np.ndarray, comp_mask: np.ndarray, top_left: Tuple[int, int], overlap_fraction: float = 0.2
) -> bool:
    """Check for excessive overlap between a given pill mask and a composite image mask.

    Determines if the pill, specified by 'mask' and positioned at 'top_left' on the composite
    image, overlaps with any pre-existing elements in the composite image more than the specified
    'overlap_fraction'. If the overlap exceeds the allowed fraction, returns False, otherwise
    returns True.

    Args:
        mask (np.ndarray): Binary mask of the pill to be added to the composition.
        comp_mask (np.ndarray): Binary mask of the pre-existing composite image.
        top_left (Tuple[int, int]): Coordinates (y, x) for the top-left corner where the
            pill should be positioned in the composition.
        overlap_fraction (float, optional): Threshold for the maximum allowed fraction of the pill
            that may overlap with pre-existing elements. Defaults to 0.2.

    Returns:
        bool: False if the overlap fraction exceeds the specified 'overlap_fraction', True otherwise.
    """
    y, x = top_left
    h_mask, w_mask = mask.shape

    # Extract a section from the composition mask that matches the pill's location and size
    comp_patch = comp_mask[x : x + h_mask, y : y + w_mask]

    # Compute overlap between the pill mask and the extracted composition patch
    h_patch, w_patch = comp_patch.shape
    overlap_patch = mask[:h_patch, :w_patch] * comp_patch

    # If the sum of the overlapped area divided by the total area of the pill is less than
    # the allowed overlap fraction, return True; else False.
    return overlap_patch.sum() / mask.sum() <= overlap_fraction
