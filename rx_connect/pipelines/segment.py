from pathlib import Path
from typing import Any, List, Union, cast

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from segment_anything import SamAutomaticMaskGenerator, build_sam
from ultralytics import YOLO

from rx_connect import CACHE_DIR, PIPELINES_DIR, SHARED_REMOTE_DIR
from rx_connect.core.types.segment import SamHqSegmentResult
from rx_connect.core.utils.sam_utils import get_best_mask
from rx_connect.pipelines.base import RxBase
from rx_connect.tools.data_tools import fetch_from_remote
from rx_connect.tools.device import get_best_available_device
from rx_connect.tools.logging import setup_logger
from rx_connect.tools.serialization import load_yaml
from rx_connect.tools.timers import timer

logger = setup_logger()


class RxSegmentation(RxBase):
    """Object wrapping segmentation model predictions."""

    _model: Union[SamAutomaticMaskGenerator, YOLO]

    def __init__(
        self,
        cfg: Union[str, Path] = f"{PIPELINES_DIR}/configs/Dev/segment_YOLO_config.yml",
    ) -> None:
        super().__init__(cfg)

    def _load_cfg(self) -> None:
        """Loads the config file and sets the attributes.
        YOLO and SAM could share same format of yaml file"""
        conf = load_yaml(self._cfg)
        self._model_type = conf.segmentation.model_type
        if self._model_type == "SAM":
            self._stability_score_thresh = conf.segmentation.stability_score_thresh
        if self._model_type == "YOLO":
            self._score_thresh = conf.segmentation.score_thresh
            self._imgsz_setting = conf.segmentation.imgsz_setting
        self._model_ckpt = conf.segmentation.model_ckpt

    def _load_model(self) -> None:
        """Loads the model from a checkpoint on the best available device. There are two options:
        1. If SAM_flag is True, then the model is loaded from HuggingFace Hub.
        2. If SAM_flag is False, then the model is loaded from a local path.
        """
        device = get_best_available_device()

        if self._model_type == "SAM":
            ckpt_path = hf_hub_download(repo_id="ybelkada/segment-anything", filename=self._model_ckpt)
            self._model = SamAutomaticMaskGenerator(
                build_sam(checkpoint=ckpt_path).to(device),
                stability_score_thresh=self._stability_score_thresh,
            )
            logger.info(f"Loaded SAM-HQ model from {ckpt_path} on {device}.")

        elif self._model_type == "YOLO":
            model_path = fetch_from_remote(self._model_ckpt, cache_dir=CACHE_DIR / "segmentation")
            self._model = YOLO(str(model_path))
            logger.info(f"Loaded YOLOv8n-seg model from {self._model_ckpt} on {device}.")

        else:
            raise ValueError(f"{self._model_type} is an unknown model type. Please use SAM or YOLO model.")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        # YOLO model will modify image
        return image.copy()

    def _predict(self, image: np.ndarray) -> Union[List[SamHqSegmentResult], np.ndarray]:
        """Predicts the segmentation of the image. The output is a list of dictionaries with
        keys: bbox, segmentation, stability_score.
        """
        if self._model_type == "SAM":
            pred = cast(SamAutomaticMaskGenerator, self._model).generate(image)
            return cast(List[SamHqSegmentResult], pred)
        else:
            # The reason to do padding: current trained model (640*640 img resolution)
            # is sensitive to scale of the pills,
            # so do padding to ensure the scale or the pill in ROI is similar
            self._img_height_setting, self._img_width_setting = self._imgsz_setting
            self._height_diff, self._width_diff = (
                self._img_height_setting - image.shape[0],
                self._img_width_setting - image.shape[1],
            )
            # for height: both top and bottom needs offset: height_diff//2
            # (might having subpixel, use minus to get the remaining pixel for another offset)
            # same for width
            self._height_offset, self._width_offset = self._height_diff // 2, self._width_diff // 2

            image = cv2.copyMakeBorder(
                image,
                self._height_offset,
                self._height_diff - self._height_offset,
                self._width_offset,
                self._width_diff - self._width_offset,
                cv2.BORDER_CONSTANT,
            )

            # result[0] is because ultralytics library way of package all the results in one list
            return cast(YOLO, self._model)(image, conf=self._score_thresh, verbose=False)[0].cpu().numpy()

    def _postprocess(self, prediction: Any) -> np.ndarray:
        """
        Convert and return the result as List[SegmentResult].

        Args:
            For SAM:
                masks (List[SamHqSegmentResult]): A list of output masks.
            For YOLO (from ultralytics library):
                An object has attributes of: boxes, masks, probs, orig_shape.
                For each attribute, it contains all stuff of that attribute.
                For bbox:
                    prediction.boxes -> it contains: type; shape; bbox
                    prediction.boxes.boxes -> it contains bbox: Nx6 (x,w,w,h,conf,class_prob).
                For mask:
                    prediction.masks -> it contains: type; shapel mask
                    prediction.masks.masks -> it contains binary mask: Nxlen(img_row)xlen(img_col)
                For confidence score:
                    prediction.boxes.conf -> it is the same value as the 5th value in prediction.boxes.boxes
                For orig_shape:
                    original shape of the image

                Also, since we do the padding and we just want the ROI mask,
                we need to crop it back to the original ROI size.

        Returns: bbox, segmented mask, and confidence score.
            For SAM:
                np.ndarray: The best mask for the fore/background separation.
            For YOLO:
                np.ndarray: The fused multi-label mask for the all pill masks.
        """
        if self._model_type == "SAM":  # the latest get_best_mask() takes original SAM results as input
            return get_best_mask(prediction).astype(np.uint8)

        else:
            # since we are doing ROI segment now, just pick the first mask (highest confidence) and discard other pillls
            # if later on we need to consider other pills, just pick the remaining masks
            predicton = prediction.masks.masks[0].astype(np.int8)
            bottom_place = self._img_height_setting - (self._height_diff - self._height_offset)
            right_place = self._img_width_setting - (self._width_diff - self._width_offset)

            return predicton[self._height_offset : bottom_place, self._width_offset : right_place]

    @timer()
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Perform full segmentation.

        Args:
            image (np.ndarray): Input image. Single image is expected.

        Returns:
            For SAM:
                np.ndarray: The best mask for the fore/background separation.
            For YOLO:
                np.ndarray: The multi-label mask.

        """
        assert image.ndim == 3, f"Image should be a 3D array, but got a {image.ndim}D array."
        image = self._preprocess(image)
        prediction = self._predict(image)
        processed_result = self._postprocess(prediction)

        return processed_result


if __name__ == "__main__":
    from rx_connect.pipelines.detection import RxDetection
    from rx_connect.pipelines.image import RxVision

    # test example
    test_image_path = (
        f"{SHARED_REMOTE_DIR}/synthetic_seg_data/dataset_4k/segmentation/dest/test/images/"
        "ffd495f4-8e83-46df-8bca-5adf7415681a.jpg"
    )

    # tested both SAM and YOLO
    config_file_SAM = f"{PIPELINES_DIR}/configs/Dev/segment_SAM_config.yml"
    config_file_YOLO = f"{PIPELINES_DIR}/configs/Dev/segment_YOLO_config.yml"

    # instantiate count object
    detection_obj = RxDetection()
    # instantiate segment object
    segmentObj = RxSegmentation(config_file_YOLO)
    # instantiate image object
    countSegmentObj = RxVision()

    # Load the image and set the counter and segmenter objects
    countSegmentObj.set_counter(detection_obj)
    countSegmentObj.set_segmenter(segmentObj)
    countSegmentObj.load_image(test_image_path)

    """Single image for all pill segmentation
    For SAM: foreground segmentation
    For YOLO: single mask with all pill having their own indices
    """
    results_mask = countSegmentObj.segmentation
    countSegmentObj.visualize_segmentation()
