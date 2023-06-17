from typing import List, TypedDict

import numpy as np


class SegmentResult(TypedDict):
    bbox: List[int]
    """Bounding box of the segment. Format: [x_min, y_min, width, height]
    """
    mask: np.ndarray
    """Segmentation mask. Format: (H, W)
    """
    score: float
    """A measure of the mask's quality.
    """


class SamHqSegmentResult(TypedDict):
    segmentation: np.ndarray
    """The segmentation mask. Format: (H, W)
    """
    bbox: List[int]
    """The box around the mask, in XYWH format.
    """
    area: int
    """The area in pixels of the mask.
    """
    predicted_iou: float
    """The model's own prediction of the mask's quality.
    """
    point_coords: List[List[int]]
    """The point coordinates input to the model to generate this mask.
    """
    stability_score: float
    """A measure of the mask's quality.
    """
    crop_box: List[int]
    """The crop of the image used to generate the mask, given in XYWH format.
    """
