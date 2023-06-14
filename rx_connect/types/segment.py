from pathlib import Path
from typing import Any, Dict, List, TypedDict, Union

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from segment_anything import SamAutomaticMaskGenerator, build_sam

from rx_connect.tools.data_tools import fetch_from_remote
from rx_connect.tools.logging import setup_logger
from rx_connect.types.image import RxImageBase, RxImageCountSegment

logger = setup_logger()


class SegmentResult(TypedDict):
    """
    The segment result should contain bbox, mask, and score
    """

    bbox: List[int]
    mask: np.ndarray
    score: float


class RxSegmentation:
    """
    RxSegmentation inference class. This class implements the functions to load model and predict.
    """

    def __init__(self, config_path: str = "./configs/Dev/segment_configs.yml") -> None:
        """
        Constructor.

        Args:
            config_path (str, optional): Path to the config file.
            Defaults to "./configs/Dev/segment_configs.yml".
        """
        self._config_path = config_path
        self._load_config()
        self._load_model()

    def _load_yaml(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Loads the yaml file.
        """
        logger.assertion(Path(config_path).exists(), f"Couldn't find config file at {config_path}.")
        with open(config_path, "r") as f:
            data_loaded = yaml.safe_load(f)
        return data_loaded

    def _load_config(self) -> None:
        """
        Reads the parameters from the config file
        """
        self._config = self._load_yaml(self._config_path)
        self._model_path = self._config["pill_segmentation"]["model_path"]
        self._model_chkpt = self._config["pill_segmentation"]["model_chkpt"]
        self._SAM_flag = self._config["pill_segmentation"]["SAM_flag"]
        self._stability_score_thresh = self._config["pill_segmentation"]["stability_score_thresh"]

    def _load_model(self) -> None:
        """
        Loads the model.
            SAM_flag True: Download SAM model online if using SAM model;
            SAM_flag False: Load our own model from server or local.
        """
        if self._SAM_flag:
            ckpt_path = hf_hub_download(self._model_path, self._model_chkpt)
            self._model = SamAutomaticMaskGenerator(
                build_sam(checkpoint=ckpt_path), stability_score_thresh=self._stability_score_thresh
            )
        else:
            self._model = torch.load(
                fetch_from_remote(self._model_path)
            )  # this line is not tested yet, need have our own model to test

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        return image

    def _predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detection network inference function.

        Args:
            image (np.ndarray): Input image.

        Returns:
            List[Result]: List of Result objects, each Result contains (boxes, masks, scores).
        """
        if self._SAM_flag:
            return self._model.generate(image)
        else:
            return self._model.predict(image)  # this line is not tested yet, need have our own model to test

    def segment(self, image: np.ndarray) -> List[SegmentResult]:
        """
        Segment function that calls the prediction and format the result.

        Args:
            image (np.ndarray): Input image. Single image is expected.

        Returns:
            List[Tuple]: List of (bbox, mask and score) pairs.
                         For SAM model, it generates multiple masks with their
                         corresponding bbox, score etc for difference instances in the image
        """
        logger.assertion(
            len(image.shape) == 3, f"Single image with RGB channels expected, but provided {image.shape}"
        )
        results = self._predict(image=image)  #
        # bbox format: [x_min, y_min, width, height]
        return [
            SegmentResult(bbox=result["bbox"], mask=result["segmentation"], score=result["stability_score"])
            for result in results
        ]


if __name__ == "__main__":
    # test example
    test_image_path = "/Users/sxiangab/Documents/synthetic3k_simple_new/images/1_17.jpg"

    # instantiate objects
    segmentObj = RxSegmentation("./configs/Dev/segment_configs.yml")
    imageObj = RxImageBase()
    imageObj.load_from_path(test_image_path)
    countSegmentObj = RxImageCountSegment(imageObj)
    countSegmentObj.set_segmenter(segmentObj)

    # Full Segmentation -> results are list of [bbox, mask, score]
    results = countSegmentObj.get_full_segmentation()
