from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from segment_anything import SamAutomaticMaskGenerator, build_sam

from rx_connect.core.types.detection import CounterModuleOutput
from rx_connect.core.types.segment import SamHqSegmentResult, SegmentResult
from rx_connect.dataset_generator.sam_utils import get_best_mask
from rx_connect.tools.data_tools import fetch_from_remote
from rx_connect.tools.logging import setup_logger
from rx_connect.tools.serialization import load_yaml
from rx_connect.tools.timers import timer
from rx_connect.types import TYPES_DIR
from rx_connect.types.base import RxBase

logger = setup_logger()


class RxSegmentation(RxBase):
    """Object wrapping segmentation model predictions."""

    def __init__(self, cfg: Union[str, Path] = f"{TYPES_DIR}/configs/Dev/segment_config.yml") -> None:
        super().__init__(cfg)

    def _load_cfg(self) -> None:
        """Loads the config file and sets the attributes."""
        conf = load_yaml(self._cfg)
        self._model_ckpt = conf.segmentation.model_ckpt
        self._SAM_flag = conf.segmentation.SAM_flag
        self._stability_score_thresh = conf.segmentation.stability_score_thresh

    def _load_model(self) -> None:
        """Loads the model. There are two options:
        1. If SAM_flag is True, then the model is loaded from HuggingFace Hub.
        2. If SAM_flag is False, then the model is loaded from a local path.
        """
        if self._SAM_flag:
            ckpt_path = hf_hub_download(repo_id="ybelkada/segment-anything", filename=self._model_ckpt)
            self._model = SamAutomaticMaskGenerator(
                build_sam(checkpoint=ckpt_path), stability_score_thresh=self._stability_score_thresh
            )
        else:
            # NOTE: This functionality hasn't been tested yet.
            self._model = torch.load(fetch_from_remote(self._model_ckpt, cache_dir=".cache/segmentation"))

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        return image

    def _predict(self, image: np.ndarray) -> List[SamHqSegmentResult]:
        """Predicts the segmentation of the image. The output is a list of dictionaries with
        keys: bbox, segmentation, stability_score.
        """
        if self._SAM_flag:
            return self._model.generate(image)
        else:
            return self._model.predict(image)  # NOTE: Not tested yet.

    def _postprocess(
        self, pred: List[SamHqSegmentResult], bbox_xyxy_list: List[List[int]]
    ) -> List[SamHqSegmentResult]:
        """
        Return the ROI of mask from the full segmented mask.
        First, select the best mask from a lisf of masks.
        Then, crop out that best mask with the provided ROI from detection module.

        Args:
            masks (List[Dict[str, Any]]): A list of masks.
            bbox (CounterModuleOutput): Bounding box from detection module.
        Returns:
            cropped masks (SegmentResult): The cropped masks from the best mask
        """
        best_seg_mask = get_best_mask(pred)

        return [best_seg_mask[y1:y2, x1:x2] for (x1, y1, x2, y2) in bbox_xyxy_list]

    def _get_raw_seg_results(self, image: np.ndarray) -> List[SamHqSegmentResult]:
        """
        Obtain raw segmentation results

        Args:
            image (np.ndarray): Input image. Single image is expected.

        Returns:
            List[SamHqSegmentResult]: List of raw SAM segmentation components.
        """
        logger.assertion(image.ndim == 3, f"Image should be a 3D array, but got a {image.ndim}D array.")
        image = self._preprocess(image)
        results = self._predict(image)

        return results

    @timer
    def segment_full(self, image: np.ndarray, bboxes: List[CounterModuleOutput]) -> List[SegmentResult]:
        """
        Return the best full segmentation mask,
        then using bbox from detection module to crop the ROI masks.

        Args:
            image (np.ndarray): Input image. Single image is expected.
            bbox (CounterModuleOutput): Bounding box from detection module.

        Returns:
                List[SegmentResult]: List of (bbox, mask, score) pairs.
        """

        results = self._get_raw_seg_results(image)

        logger.assertion(bboxes is not None, "bboxes are not available.")
        bbox_xyxy_list = [item.bbox for item in bboxes]
        bbox_score_list = [item.scores for item in bboxes]

        output_results = self._postprocess(results, bbox_xyxy_list)

        return [
            SegmentResult(bbox=bbox, mask=mask, score=score)
            for bbox, mask, score in zip(bbox_xyxy_list, output_results, bbox_score_list)
        ]

    def segment_ROI(self, image: np.ndarray) -> List[SegmentResult]:
        """
        Returns all the segmentation masks for an input image (ROI) with
        their corresponding bbox, score, etc.

        Args:
            image (np.ndarray): Input image. Single image is expected.

        Returns:
            List[SegmentResult]: List of (bbox, mask and score) pairs.
        """
        results = self._get_raw_seg_results(image)

        return [
            SegmentResult(bbox=result["bbox"], mask=result["segmentation"], score=result["stability_score"])
            for result in results
        ]


if __name__ == "__main__":
    from rx_connect.types.detection import RxDetection
    from rx_connect.types.image import RxImageBase, RxImageCountSegment

    # test example
    test_image_path = "/Users/sxiangab/Documents/synthetic3k_simple_new/images/1_17.jpg"

    # instantiate image object
    imageObj = RxImageBase()
    imageObj.load_image(test_image_path)

    # instantiate count object
    detection_obj = RxDetection()

    # instantiate objects
    segmentObj = RxSegmentation()
    countSegmentObj = RxImageCountSegment(imageObj)
    countSegmentObj.set_counter(detection_obj)
    countSegmentObj.set_segmenter(segmentObj)

    # Full Segmentation -> results are list of [bbox, mask, score]
    results = countSegmentObj.full_segmentation
