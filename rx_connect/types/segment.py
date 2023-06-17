from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from segment_anything import SamAutomaticMaskGenerator, build_sam

from rx_connect.core.types.segment import SamHqSegmentResult, SegmentResult
from rx_connect.tools.data_tools import fetch_from_remote
from rx_connect.tools.logging import setup_logger
from rx_connect.tools.serialization import load_yaml
from rx_connect.types import TYPES_DIR
from rx_connect.types.base import RxBase
from rx_connect.types.image import RxImageBase, RxImageCountSegment

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

    def _postprocess(self, pred: List[SamHqSegmentResult]) -> List[SamHqSegmentResult]:
        return pred

    def segment(self, image: np.ndarray) -> List[SegmentResult]:
        """Returns all the segmentation masks for an input image with their corresponding bbox,
        score, etc.
        """
        logger.assertion(image.ndim == 3, f"Image should be a 3D array, but got a {image.ndim}D array.")
        image = self._preprocess(image)
        results = self._predict(image)
        output_results = self._postprocess(results)

        return [
            SegmentResult(bbox=result["bbox"], mask=result["segmentation"], score=result["stability_score"])
            for result in output_results
        ]


if __name__ == "__main__":
    # test example
    test_image_path = "/Users/sxiangab/Documents/synthetic3k_simple_new/images/1_17.jpg"

    # instantiate objects
    segmentObj = RxSegmentation()
    imageObj = RxImageBase()
    imageObj.load_from_path(test_image_path)
    countSegmentObj = RxImageCountSegment(imageObj)
    countSegmentObj.set_segmenter(segmentObj)

    # Full Segmentation -> results are list of [bbox, mask, score]
    results = countSegmentObj.get_full_segmentation()
