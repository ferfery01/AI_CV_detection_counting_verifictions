from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from rx_connect.dataset_generator.transform import resize_bg


def load_pill_mask_paths(pill_mask_dir: Path) -> List[Tuple[Path, Path]]:
    """Load the pill image and the corresponding mask paths.

    Args:
        pill_mask_dir: The directory containing the pill images and masks.

    Returns:
        pill_mask_paths: The paths to the pill image and mask.
    """
    pill_mask_paths: List[Tuple[Path, Path]] = [
        (p, pill_mask_dir / "masks" / p.name) for p in (pill_mask_dir / "images").glob("*.jpg")
    ]
    return pill_mask_paths


def get_img_and_mask(pill_mask_paths: Tuple[Path, Path], thresh: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """Get the image and mask from the pill mask paths.

    Args:
        pill_mask_paths: The paths to the pill image and mask.
        thresh: The threshold at which to binarize the mask.

    Returns:
        img: The pill image.
        mask: The pill mask.
    """
    img_path, mask_path = pill_mask_paths
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask[mask <= thresh] = 0
        mask[mask > thresh] = 1
    except Exception as e:
        raise FileNotFoundError(
            f"Could not find mask for image {img_path.name}. Did you run `python -m countpillar/generate_masks.py`?"
        ) from e

    return img, mask


def load_bg_image(path: Path, min_dim: int, max_dim) -> np.ndarray:
    """Load and resize the background image.

    Args:
        path: The path to the background image.
        min_dim: The minimum dimension of the background image.
        max_dim: The maximum dimension of the background image.

    Returns:
        bg_img: The background image as a numpy array.
    """
    bg_img = cv2.imread(str(path))
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    bg_img = resize_bg(bg_img, max_dim, min_dim)

    return bg_img


def create_yolo_annotations(mask_comp: np.ndarray, labels_comp: List[int]) -> List[List[float]]:
    """Create YOLO annotations from a composition mask.

    Args:
        mask_comp: The composition mask.
        labels_comp: The labels of the composition.

    Returns:
        annotations_yolo: The YOLO annotations.
    """
    comp_w, comp_h = mask_comp.shape[1], mask_comp.shape[0]

    obj_ids = np.unique(mask_comp).astype(np.uint8)[1:]
    masks = mask_comp == obj_ids[:, None, None]

    annotations_yolo: List[List[float]] = []
    for i in range(len(labels_comp)):
        pos = np.where(masks[i])
        xmin, xmax = np.min(pos[1]), np.max(pos[1])
        ymin, ymax = np.min(pos[0]), np.max(pos[0])

        xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
        w, h = xmax - xmin, ymax - ymin

        annotations_yolo.append(
            [
                labels_comp[i] - 1,
                round(float(xc / comp_w), 5),
                round(float(yc / comp_h), 5),
                round(float(w / comp_w), 5),
                round(float(h / comp_h), 5),
            ]
        )

    return annotations_yolo
