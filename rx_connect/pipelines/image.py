from pathlib import Path
from typing import List, Optional, Union, cast

import cv2
import numpy as np
import torch
import torchshow as ts
from PIL import Image
from skimage import io

from rx_connect.core.images.visualize import visualize_gallery
from rx_connect.core.types.detection import CounterModuleOutput
from rx_connect.pipelines.detection import RxDetection
from rx_connect.pipelines.generator import RxImageGenerator
from rx_connect.pipelines.segment import RxSegmentation
from rx_connect.pipelines.vectorizer import RxVectorizer
from rx_connect.tools.data_tools import fetch_from_remote
from rx_connect.tools.logging import setup_logger

logger = setup_logger()


class RxVisionBase:
    """Base class for RxVision objects. This class implements all the different loading methods."""

    _image: Optional[np.ndarray] = None
    _ref_image: Optional[np.ndarray] = None

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        """
        Clear any results from previous runs.
        Called when old results becomes invalidated; e.g. new image loaded, new reference image,
        new processing tool objects...etc.
        """
        pass

    def load_from_camera(self) -> None:
        """Loads the image from default camera."""
        self._reset()
        _, self._image = cv2.VideoCapture(0).read()

    def load_image(self, image: Union[np.ndarray, torch.Tensor, Image.Image, str, Path]) -> None:
        """Set the tray-view image."""
        self._reset()
        self._image = self._image_loader(image)

    def load_ref_image(self, image: Union[np.ndarray, torch.Tensor, Image.Image, str, Path]) -> None:
        """Set the reference image."""
        self._reset()
        self._ref_image = self._image_loader(image)

    def _image_loader(
        self, image: Union[np.ndarray, torch.Tensor, Image.Image, str, Path]
    ) -> Optional[np.ndarray]:
        """Loads the image from the given image object. The image object can be a numpy array,
        torch tensor, PIL image, or a path to an image. The path can be a local path or a remote
        path. If the path is remote, the image is downloaded to the cache directory. The image is
        loaded as a numpy array.

        Args:
            image (Union[np.ndarray, torch.Tensor, Image]): Image as image object.

        Note:
            If the image is str/Path that leads to a '.png' file, there might be a 4th channel that will be ignored.
        """
        if isinstance(image, np.ndarray):
            return image
        elif isinstance(image, torch.Tensor):
            return image.numpy()
        elif isinstance(image, Image.Image):
            return np.array(image)
        elif isinstance(image, (str, Path)):
            image_path = fetch_from_remote(image)
            return io.imread(image_path)[:, :, :3]
        else:
            raise TypeError(f"Image type {type(image)} not supported.")

    def load_from_generator(self, generator_obj: RxImageGenerator, **kwargs: bool) -> None:
        """Loads the image from the given generator object.

        Args:
            generator_obj (Generator): Image Generator object.
            kwargs: Keyword arguments to be passed to the generator object.
        """
        self._reset()
        self._image, _, _, self._gt_bbox, _ = generator_obj.generate(**kwargs)
        self.load_ref_image(generator_obj.reference_pills[0])

    def visualize(self) -> None:
        """Utility function to visualize image stored in the object."""
        ts.show([self.ref_image, self.image])

    @property
    def ref_image(self) -> np.ndarray:
        """Access the reference image property.
        Check if the reference image has been loaded, and return the reference image.

        Returns:
            np.ndarray: Loaded reference image.
        """
        assert self._ref_image is not None, "Reference image not loaded."
        return self._ref_image

    @property
    def image(self) -> np.ndarray:
        """Access the image property.
        Check if the image has been loaded, and return the image.

        Returns:
            np.ndarray: Loaded image.
        """
        assert self._image is not None, "Image not loaded."
        return self._image

    @property
    def gt_ROIs(self) -> List[np.ndarray]:
        """Return the cropped image views as the regions of interest (ROIs).

        Returns:
            List[np.ndarray]: List of cropped image views.
        """
        assert hasattr(self, "_gt_bbox"), "Groundtruth bounding box information is not available."
        return [self.image[xmin:xmax, ymin:ymax] for (xmin, xmax, ymin, ymax) in self._gt_bbox]


class RxVisionDetect(RxVisionBase):
    """Vision class for detecting all the pills in an image. This class inherits RxVisionBase for
    all the loading methods.
    """

    _bounding_boxes: Optional[List[CounterModuleOutput]] = None

    def __init__(self) -> None:
        super().__init__()
        self._counterObj: Optional[RxDetection] = None

    def _reset(self) -> None:
        super()._reset()
        self._bounding_boxes = None

    def set_counter(self, counterObj: RxDetection) -> None:
        """Sets the counter object. Reset any existing results to None when there's a new counter.

        Args:
            counterObj (Counter): Counter object.
        """
        self._counterObj = counterObj
        self._reset()

    def visualize_ROIs(self, img_per_row: int = 5, labels: Optional[List[str]] = None) -> None:
        """Utility function to visualize cropped ROIs.

        First change the list into a list of list, then visualize.

        Args:
            img_per_row (int): The number of images per row.
            labels (list[str]): The labels to show with the ROIs.
        """

        def label_img(img: np.ndarray, label: Optional[str]) -> np.ndarray:
            if label is None:
                return img
            labeled_img = np.full(
                (img.shape[0] + 50, img.shape[1], img.shape[2]),
                (0, 255, 255),
                dtype=img.dtype,
            )
            labeled_img[50:, :, :] = img
            cv2.putText(labeled_img, label, (5, 40), cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 0, 0), thickness=2)
            return labeled_img

        show_imgs = self.ROIs
        if labels is not None:
            show_imgs = [label_img(img, labels[i]) for i, img in enumerate(show_imgs)]

        visualize_gallery(show_imgs, img_per_row=img_per_row)

    def draw_bounding_boxes(self) -> np.ndarray:
        """Utility function to draw bounding boxes on the image."""
        img_bb = self.image.copy()
        for (x1, y1, x2, y2), _ in self.bounding_boxes:
            cv2.rectangle(img_bb, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return img_bb

    def visualize_bounding_boxes(self) -> None:
        """Utility function to visualize bounding boxes found."""
        img_bb = self.draw_bounding_boxes()
        ts.show(img_bb)

    @property
    def bounding_boxes(self) -> List[CounterModuleOutput]:
        """Access the bounding boxs property.
        Check if the bounding boxes has been produced. If not, count before returning them.

        Returns:
            List[CounterModuleOutput]: Bounding boxes. Format: [((X1, Y1, X2, Y2), score), ...]
        """
        if self._bounding_boxes is None:
            assert self._counterObj is not None, "Counter object not set."
            self._bounding_boxes = self._counterObj.count(self.image)
        return self._bounding_boxes

    @property
    def pill_count(self) -> int:
        """Return the length of the bounding boxes as the count.

        Returns:
            int: Count.
        """
        return len(self.bounding_boxes)

    @property
    def ROIs(self) -> List[np.ndarray]:
        """Return the cropped image views as the regions of interest (ROIs).

        Returns:
            List[np.ndarray]: List of cropped image views.
        """
        return [self.image[y1:y2, x1:x2] for (x1, y1, x2, y2), _ in self.bounding_boxes]


class RxVisionSegment(RxVisionDetect):
    """Vision class for segmenting all the pills in an image. This class inherits RxVisionDetect for all
    the loading and detection methods.
    """

    _full_mask: Optional[np.ndarray] = None
    _masked_ROIs: Optional[List[np.ndarray]] = None
    _ROI_masks: Optional[List[np.ndarray]] = None

    def __init__(self) -> None:
        super().__init__()
        self._segmenterObj: Optional[RxSegmentation] = None

    def _reset(self) -> None:
        super()._reset()
        self._full_mask = None
        self._masked_ROIs = None
        self._ROI_masks = None

    def set_segmenter(self, segmenterObj: RxSegmentation) -> None:
        """Sets the segmenter object. Reset any existing results to None when there's a new segmenter.

        Args:
            segmenterObj (Segmenter): Segmenter object.
        """
        self._segmenterObj = segmenterObj
        self._reset()

    def visualize_full_segmentation(self) -> None:
        """Visualizd the full segment results (one single binary mask).
        For now, only call it when using SAM for full segment.
        """
        ts.show(self.full_segmentation)

    def visualize_ROI_segmentation(self, img_per_row: int = 5) -> None:
        """Visualize the segment results (binary masks for each ROI)."""
        visualize_gallery(self.ROI_segmentation, img_per_row=img_per_row)

    def visualize_masked_ROIs(self, img_per_row: int = 5) -> None:
        """Visualize the masked ROIs."""
        visualize_gallery(self.masked_ROIs, img_per_row=img_per_row)

    @property
    def full_segmentation(self) -> np.ndarray:
        """
        Run full segmentation with SAM to separate forground/background.
        From the segmentation results, choose the one containing all pills.

        Returns:
            np.ndarray: The mask that separates the background and all the pills.
        """
        if self._full_mask is None:
            assert self._segmenterObj is not None, "Segmenter object not set."
            assert self._segmenterObj._model_type == "SAM", "Not using SAM for full seg."
            mask = self._segmenterObj.segment(self.image)
            # x = column, y = row, so need to reverse the order
            self._full_mask = cv2.resize(mask, self.image.shape[:2][::-1])

        return self._full_mask

    @property
    def masked_ROIs(self) -> List[np.ndarray]:
        """
        Return the masked ROIs.

        Returns:
            List[np.ndarray]: The ROIs with background removed.
        """
        if self._masked_ROIs is None:
            self._masked_ROIs = cast(
                List[np.ndarray],
                [cv2.bitwise_or(ROI, ROI, mask=mask) for ROI, mask in zip(self.ROIs, self.ROI_segmentation)],
            )
        return self._masked_ROIs

    @property
    def ROI_segmentation(self) -> List[np.ndarray]:
        """Check if the ROI segmentation has been produced. If not, segment before returning it.
        If using SAM, crop the ROI masks from the full segmentation mask, using the bbox from detection module.
        If using YOLO, directly segment from the ROI image.

        Returns:
            List[np.ndarray]: Segmentation results for each ROI.
        """
        if self._ROI_masks is None:
            assert self._segmenterObj is not None, "Segmenter object not set."
            if self._segmenterObj._model_type == "SAM":
                bbox_xyxy_list = [item.bbox for item in self.bounding_boxes]
                self._ROI_masks = [
                    self.full_segmentation[y1:y2, x1:x2] for (x1, y1, x2, y2) in bbox_xyxy_list
                ]
            else:  # YOLO
                self._ROI_masks = [self._segmenterObj.segment(ROI) for ROI in self.ROIs]

        return self._ROI_masks


class RxVisionVerify(RxVisionSegment):
    """Vision class for verification methods. This class inherits from RxVisionSegment for the
    loading, detection, and segmentation methods.
    """

    _vectorized_ROIs: Optional[List[np.ndarray]] = None
    _vectorized_ref: Optional[np.ndarray] = None
    _similarity_scores: Optional[np.ndarray] = None

    def __init__(self) -> None:
        super().__init__()
        self._vectorizerObj: Optional[RxVectorizer] = None

    def _reset(self) -> None:
        super()._reset()
        self._vectorized_ROIs = None
        self._vectorized_ref = None
        self._similarity_scores = None

    def set_vectorizer(self, vectorizerObj: RxVectorizer) -> None:
        """Sets the vectorizer object. Reset any existing results to None when there's a new vectorizer.

        Args:
            vectorizerObj (vectorizer): Vectorizer object.
        """
        self._vectorizerObj = vectorizerObj
        self._reset()

    def visualize_similarity_scores(self, img_per_col: int = 5) -> None:
        """Utility function to visualize similarity scores along with ROIs.

        Args:
            img_per_col (int): The number of images per column.
        """
        labels = [f"{i}: {score*100:.1f}%" for i, score in enumerate(self.similarity_scores)]
        self.visualize_ROIs(img_per_col, labels=labels)

    @property
    def vectorized_ref(self) -> np.ndarray:
        """Access property of the vectorized reference image; vectorize it if not already.

        Returns:
            np.ndarray: The vectorized reference image.
        """
        if self._vectorized_ref is None:
            assert self._vectorizerObj is not None, "Vectorizer object not set."
            self._vectorized_ref = self._vectorizerObj.encode(self.ref_image)
        return self._vectorized_ref

    @property
    def vectorized_ROIs(self) -> List[np.ndarray]:
        """Access property of the vectorized ROIs; vectorize them if not already.

        Returns:
            List[np.ndarray]: The vectorized ROIs.
        """
        if self._vectorized_ROIs is None:
            assert self._vectorizerObj is not None, "Vectorizer object not set."
            self._vectorized_ROIs = self._vectorizerObj.encode(
                self.masked_ROIs if self._vectorizerObj.require_masked_input else self.ROIs
            )
        return self._vectorized_ROIs

    @property
    def similarity_scores(self) -> np.ndarray:
        """Returns the similarity for all the ROIs."""
        if self._similarity_scores is None:
            assert self._vectorizerObj is not None, "Vectorizer object not set."
            self._similarity_scores = self._vectorizerObj._similarity_fn(
                [self.vectorized_ref], self.vectorized_ROIs
            ).ravel()
        return self._similarity_scores


class RxVision(RxVisionVerify):
    """The Vision class encapsulates the essential methods for processing and analyzing image data related to
    pill detection and segmentation.

    Key Features:
    - Image Loading: Provides functionality for efficiently loading image files.
    - Pill Detection: Implements sophisticated techniques to accurately detect pills within an image.
    - Pill Segmentation: Offers tools for segmenting detected pills for further analysis or manipulation.
    - Pill Verification: Includes methods for comparing and verifying each detected pill against a reference image.
        This allows for identification and validation of the pill.
    """

    def __init__(self) -> None:
        super().__init__()
