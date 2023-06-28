from typing import List

import numpy as np


def create_yolo_annotations(mask_comp: np.ndarray, labels_comp: List[int]) -> List[List[float]]:
    """Create YOLO annotations from a composition mask.

    Args:
        mask_comp: The composition mask.
        labels_comp: The labels of the composition.

    Returns:
        annotations_yolo: The YOLO annotations.
        Output format: XYXY_LABEL (x1, y1, x2, y2, class_id)
    """
    comp_w, comp_h = mask_comp.shape[1], mask_comp.shape[0]

    obj_ids = np.unique(mask_comp).astype(np.uint8)[1:]
    masks = mask_comp == obj_ids[:, None, None]

    annotations_yolo: List[List[float]] = []
    for _, mask in enumerate(masks):
        pos = np.where(mask)
        xmin, xmax = np.min(pos[1]) - 1, np.max(pos[1]) + 1
        ymin, ymax = np.min(pos[0]) - 1, np.max(pos[0]) + 1

        xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
        w, h = xmax - xmin, ymax - ymin

        annotations_yolo.append(
            [
                0,  # class_id for all pills is 0
                round(float(xc / comp_w), 5),
                round(float(yc / comp_h), 5),
                round(float(w / comp_w), 5),
                round(float(h / comp_h), 5),
            ]
        )

    return annotations_yolo
