from pathlib import Path
from typing import List, NamedTuple

import numpy as np

YOLO_LABELS = "labels"
SEGMENTATION_LABELS = "comp_masks"
COCO_LABELS = "COCO"


class PillMaskPaths(NamedTuple):
    """The paths to the pill image and mask."""

    imgs_path: List[Path]
    masks_path: List[Path]


class PillMask(NamedTuple):
    """The pill image and mask."""

    image: np.ndarray
    mask: np.ndarray
