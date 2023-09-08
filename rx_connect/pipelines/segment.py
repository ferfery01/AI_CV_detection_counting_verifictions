from pathlib import Path
from typing import Any, List, Optional, Tuple, Union, cast

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms.functional as TF
from huggingface_hub import hf_hub_download
from segment_anything import SamAutomaticMaskGenerator, build_sam
from segmentation_models_pytorch.base import SegmentationModel
from ultralytics import YOLO

from rx_connect import CACHE_DIR, PIPELINES_DIR, SHARED_REMOTE_DIR
from rx_connect.core.images.masks import refine_mask
from rx_connect.core.images.types import img_to_tensor
from rx_connect.core.types.segment import SamHqSegmentResult
from rx_connect.core.utils.func_utils import to_tuple
from rx_connect.core.utils.sam_utils import get_best_mask
from rx_connect.pipelines.base import RxBase
from rx_connect.segmentation.semantic.augments import SegmentTransform
from rx_connect.tools.data_tools import fetch_from_remote
from rx_connect.tools.logging import setup_logger
from rx_connect.tools.serialization import load_yaml
from rx_connect.tools.timers import timer

logger = setup_logger()


class RxSegmentation(RxBase):
    """Object wrapping segmentation model predictions."""

    _model: Union[SamAutomaticMaskGenerator, YOLO]

    def __init__(
        self,
        cfg: Union[str, Path] = f"{PIPELINES_DIR}/configs/Dev/instance/segment_YOLO_config.yml",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(cfg, device)

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

        if self._model_type == "SAM":
            ckpt_path = hf_hub_download(repo_id="ybelkada/segment-anything", filename=self._model_ckpt)
            self._model = SamAutomaticMaskGenerator(
                build_sam(checkpoint=ckpt_path).to(self._device),
                stability_score_thresh=self._stability_score_thresh,
            )
            logger.info(f"Loaded SAM-HQ model from {ckpt_path} on {self._device}.")

        elif self._model_type == "YOLO":
            model_path = fetch_from_remote(self._model_ckpt, cache_dir=CACHE_DIR / "segmentation")
            self._model = YOLO(str(model_path))
            logger.info(f"Loaded YOLOv8n-seg model from {self._model_ckpt} on {self._device}.")

        else:
            raise ValueError(f"{self._model_type} is an unknown model type. Please use SAM or YOLO model.")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        # YOLO model will modify image
        return image.copy()

    def _predict(self, image: Union[np.ndarray, torch.Tensor]) -> Union[List[SamHqSegmentResult], np.ndarray]:
        """Predicts the segmentation of the image. The output is a list of dictionaries with
        keys: bbox, segmentation, stability_score.
        """
        image = np.array(image) if isinstance(image, torch.Tensor) else image
        if self._model_type == "SAM":
            pred = cast(SamAutomaticMaskGenerator, self._model).generate(image)
            return cast(List[SamHqSegmentResult], pred)
        else:
            # The reason to do padding: current trained model (640*640 img resolution)
            # is sensitive to scale of the pills,
            # so do padding to ensure the scale or the pill in ROI is similar
            self._img_height_setting, self._img_width_setting = self._imgsz_setting
            self._img_height, self._img_width = image.shape[:2]
            self._height_diff, self._width_diff = (
                self._img_height_setting - self._img_height,
                self._img_width_setting - self._img_width,
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

            # the region needs to be cropped back to the original size
            bottom_place = self._img_height_setting - (self._height_diff - self._height_offset)
            right_place = self._img_width_setting - (self._width_diff - self._width_offset)

            try:
                prediction = prediction.masks.masks[0].astype(np.int8)
            except AttributeError:
                # if the prediction is none, generate an all-one mask
                prediction = np.ones((self._img_height_setting, self._img_width_setting), np.int8)

            return prediction[self._height_offset : bottom_place, self._width_offset : right_place]

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
        return self(image)


class RxSemanticSegmentation(RxBase):
    def __init__(
        self,
        cfg: Union[str, Path] = f"{PIPELINES_DIR}/configs/Dev/semantic/deeplabv3_plus-resnet50.yaml",
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__(cfg, device)
        self._image_size: Optional[Tuple[int, int]] = None
        self._test_image_size: Optional[Tuple[int, int]] = None

    @property
    def transform(self) -> SegmentTransform:
        """Returns the transform to use for the semantic segmentation model inference."""
        return SegmentTransform(train=False, normalize=True, image_size=self.image_size)

    @property
    def image_size(self) -> Tuple[int, int]:
        """Returns the image size to use for the semantic segmentation model inference. If the image size
        is not set, then the image size from the config file is returned.
        """
        if self._image_size is None:
            return (self.conf.image_size[0], self.conf.image_size[1])
        return self._image_size

    @property
    def test_image_size(self) -> Tuple[int, int]:
        """Returns the image size of the test image. This is used to resize the segmentation mask to the
        original image size after inference. This is set in the `_preprocess` method.
        """
        if self._test_image_size is None:
            raise TypeError(
                f"The `{self.__class__.__name__}._test_image_size` reference has not been set yet."
            )
        return self._test_image_size

    @test_image_size.setter
    def test_image_size(self, image_size: Tuple[int, int]) -> None:
        self._test_image_size = image_size

    def _load_cfg(self) -> None:
        """Load the configuration file for the semantic segmentation model."""
        conf = load_yaml(self._cfg)
        self.conf = conf.semantic_segmentation
        self._seg_model = self.conf.model
        self._arch = self.conf.arch

        # Check if the model is available in segmentation_models_pytorch
        if not hasattr(smp, self._seg_model):
            raise ValueError(f"Model {self._seg_model} not found in `segmentation_models_pytorch`.")

    def _load_model(self) -> None:
        """Load the semantic segmentation model from the path specified in the config file."""
        _model_path = fetch_from_remote(
            self.conf.model_path, cache_dir=CACHE_DIR / "segmentation" / "semantics" / self._seg_model
        )
        if not _model_path.exists():
            raise FileNotFoundError(f"Model path {_model_path} does not exist.")

        seg_model = cast(SegmentationModel, getattr(smp, self._seg_model))
        self._model = seg_model(encoder_name=self._arch, encoder_weights=None)
        model_state_dict = torch.load(_model_path, map_location=self._device)
        self._model.load_state_dict(model_state_dict)
        self._model.eval()
        logger.info(f"Loaded {self._seg_model} model from {_model_path} on {self._device}.")

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess the image for inference."""
        self.test_image_size = to_tuple(image.shape[:2])
        return self.transform(image=image, mask=None).to(self._device)

    @torch.inference_mode()
    def _predict(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Predict the segmentation mask for the input image. The logits are passed through a sigmoid
        function to get the probability mask. The probability mask is then thresholded at 0.5 to get
        the final segmentation mask.
        """
        image = img_to_tensor(image) if isinstance(image, np.ndarray) else image
        logits_mask = self._model(image.unsqueeze(0))
        prob_mask = logits_mask.sigmoid()
        return (prob_mask > 0.5).float()

    def _postprocess(self, mask: torch.Tensor, **kwargs: int) -> np.ndarray:
        """Postprocess the mask to get the final segmentation mask. The mask is resized to the original
        image size and then refined using the `refine_mask` function. The refined mask returned is a binary
        mask.
        """
        mask = TF.resize(mask, size=self.test_image_size)
        mask_np = mask.squeeze().cpu().numpy()
        mask_np = refine_mask(mask_np, **kwargs)
        return mask_np


if __name__ == "__main__":
    from rx_connect.pipelines.detection import RxDetection
    from rx_connect.pipelines.image import RxVision

    # test example
    test_image_path = (
        f"{SHARED_REMOTE_DIR}/synthetic_seg_data/dataset_4k/segmentation/dest/test/images/"
        "ffd495f4-8e83-46df-8bca-5adf7415681a.jpg"
    )

    # tested both SAM and YOLO
    config_file_SAM = f"{PIPELINES_DIR}/configs/Dev/instance/segment_SAM_config.yml"
    config_file_YOLO = f"{PIPELINES_DIR}/configs/Dev/instance/segment_YOLO_config.yml"

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
    results_mask = countSegmentObj.ROI_segmentation
    countSegmentObj.visualize_ROI_segmentation()
