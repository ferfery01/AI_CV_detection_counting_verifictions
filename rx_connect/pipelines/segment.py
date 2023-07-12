from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from segment_anything import SamAutomaticMaskGenerator, build_sam

from rx_connect import CACHE_DIR, PIPELINES_DIR
from rx_connect.core.types.segment import SamHqSegmentResult, SegmentResult
from rx_connect.pipelines.base import RxBase
from rx_connect.tools.data_tools import fetch_from_remote
from rx_connect.tools.device import get_best_available_device
from rx_connect.tools.logging import setup_logger
from rx_connect.tools.serialization import load_yaml
from rx_connect.tools.timers import timer

logger = setup_logger()


class RxSegmentation(RxBase):
    """Object wrapping segmentation model predictions."""

    def __init__(self, cfg: Union[str, Path] = f"{PIPELINES_DIR}/configs/Dev/segment_config.yml") -> None:
        super().__init__(cfg)

    def _load_cfg(self) -> None:
        """Loads the config file and sets the attributes."""
        conf = load_yaml(self._cfg)
        self._model_ckpt = conf.segmentation.model_ckpt
        self._SAM_flag = conf.segmentation.SAM_flag
        self._stability_score_thresh = conf.segmentation.stability_score_thresh

    def _load_model(self) -> None:
        """Loads the model from a checkpoint on the best available device. There are two options:
        1. If SAM_flag is True, then the model is loaded from HuggingFace Hub.
        2. If SAM_flag is False, then the model is loaded from a local path.
        """
        if self._SAM_flag:
            ckpt_path = hf_hub_download(repo_id="ybelkada/segment-anything", filename=self._model_ckpt)
            device = get_best_available_device()
            self._model = SamAutomaticMaskGenerator(
                build_sam(checkpoint=ckpt_path).to(device),
                stability_score_thresh=self._stability_score_thresh,
            )
            logger.info(f"Loaded SAM-HQ model from {ckpt_path} on {device}.")
        else:
            # NOTE: This functionality hasn't been tested yet.
            self._model = torch.load(
                fetch_from_remote(self._model_ckpt, cache_dir=CACHE_DIR / "segmentation")
            )

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

    def _postprocess(self, prediction: List[SamHqSegmentResult]) -> List[SegmentResult]:
        """
        Convert and return the result as List[SegmentResult].

        Args:
            masks (List[SamHqSegmentResult]): A list of output masks.

        Returns:
            List[SegmentResult]: The best mask for the fore/background separation.
        """
        return [
            SegmentResult(bbox=result["bbox"], mask=result["segmentation"], score=result["stability_score"])
            for result in prediction
        ]

    @timer
    def segment_full(self, image: np.ndarray) -> List[SamHqSegmentResult]:
        """
        Perform full segmentation.

        Args:
            image (np.ndarray): Input image. Single image is expected.

        Returns:
            List[SamHqSegmentResult]: List of raw SAM segmentation components.
        """
        logger.assertion(image.ndim == 3, f"Image should be a 3D array, but got a {image.ndim}D array.")
        image = self._preprocess(image)
        results = self._predict(image)

        return results

    def segment_ROI(self, image: np.ndarray) -> List[SegmentResult]:
        """
        Returns all the segmentation masks for an input image (ROI) with
        their corresponding bbox, score, etc.

        Args:
            image (np.ndarray): Input image. Single image is expected.

        Returns:
            List[SegmentResult]: List of (bbox, mask and score) pairs.
        """
        raw_result = self.segment_full(image)
        processed_result = self._postprocess(raw_result)

        return processed_result


if __name__ == "__main__":
    from rx_connect.pipelines.detection import RxDetection
    from rx_connect.pipelines.image import RxImage

    # test example
    test_image_path = "/media/RxConnectShared/ePillID/pills/images/50090-0762_0_0.jpg"

    # instantiate count object
    detection_obj = RxDetection()

    # instantiate segment object
    segmentObj = RxSegmentation()

    # instantiate image object
    countSegmentObj = RxImage()

    # Load the image and set the counter and segmenter objects
    countSegmentObj.load_image(test_image_path)
    countSegmentObj.set_counter(detection_obj)
    countSegmentObj.set_segmenter(segmentObj)

    # Current segmentation with SAM separates forground/background
    results = countSegmentObj.background_segment
    logger.assertion(
        results.shape == countSegmentObj.image.shape[:2], "Output mask shape mismatches the input shape."
    )
