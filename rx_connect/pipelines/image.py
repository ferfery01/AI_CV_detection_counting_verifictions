from pathlib import Path
from typing import Dict, List, Optional, Union, cast

import cv2
import numpy as np
import torch
import torchshow as ts
from PIL import Image
from skimage import io
from sklearn.metrics.pairwise import cosine_similarity

from rx_connect.core.types.detection import CounterModuleOutput
from rx_connect.core.types.segment import SegmentResult
from rx_connect.core.utils.sam_utils import get_best_mask
from rx_connect.pipelines.detection import RxDetection
from rx_connect.pipelines.generator import RxImageGenerator
from rx_connect.pipelines.segment import RxSegmentation
from rx_connect.pipelines.vectorizer import RxVectorizer
from rx_connect.tools.data_tools import fetch_from_remote
from rx_connect.tools.logging import setup_logger

logger = setup_logger()


class RxImageBase:
    """Base class for RxImage objects. This class implements the loading methods."""

    def __init__(self) -> None:
        self._image: Optional[np.ndarray] = None
        self._ref_image: Optional[np.ndarray] = None

    def load_from_camera(self) -> None:
        """Loads the image from default camera."""
        _, self._image = cv2.VideoCapture(0).read()

    def load_image(self, image: Union[np.ndarray, torch.Tensor, Image.Image, str, Path]) -> None:
        """Loads the image from the given image object. The image object can be a numpy array,
        torch tensor, PIL image, or a path to an image. The path can be a local path or a remote
        path. If the path is remote, the image is downloaded to the cache directory. The image is
        loaded as a numpy array.

        Args:
            image (Union[np.ndarray, torch.Tensor, Image]): Image as image object.
        """
        if isinstance(image, np.ndarray):
            self._image = image
        elif isinstance(image, torch.Tensor):
            self._image = image.numpy()
        elif isinstance(image, Image.Image):
            self._image = np.array(image)
        elif isinstance(image, (str, Path)):
            image_path = fetch_from_remote(image)
            self._image = io.imread(image_path)
        else:
            raise TypeError(f"Image type {type(image)} not supported.")

    def load_from_generator(self, generator_obj: RxImageGenerator, **kwargs: Dict[str, bool]) -> None:
        """Loads the image from the given generator object.

        Args:
            generator_obj (Generator): Image Generator object.
            kwargs: Keyword arguments to be passed to the generator object.
        """
        self._image, *_ = generator_obj.generate(**kwargs)
        self.load_ref_image(generator_obj.reference_pills[0])

    def load_ref_image(self, ref_image: np.ndarray) -> None:
        """Set the reference image."""
        self._ref_image = ref_image

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
        logger.assertion(self._ref_image is not None, "Reference image not loaded.")
        return cast(np.ndarray, self._ref_image)

    @property
    def image(self) -> np.ndarray:
        """Access the image property.
        Check if the image has been loaded, and return the image.

        Returns:
            np.ndarray: Loaded image.
        """
        logger.assertion(self._image is not None, "Image not loaded.")  # type: ignore
        return cast(np.ndarray, self._image)


class RxImageCount(RxImageBase):
    """Image class for counting methods. This class inherits from RxImageBase for the loading methods."""

    def __init__(self) -> None:
        super().__init__()
        self._bounding_boxes: Optional[List[CounterModuleOutput]] = None
        self._counterObj: Optional[RxDetection] = None

    def set_counter(self, counterObj: RxDetection) -> None:
        """Sets the counter object. Reset any existing results to None when there's a new counter.

        Args:
            counterObj (Counter): Counter object.
        """
        self._counterObj = counterObj
        self._bounding_boxes = None

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

        show_img_reordered = [
            show_imgs[i * img_per_row : i * img_per_row + img_per_row]
            for i in range(-(len(show_imgs) // -img_per_row))
        ]
        ts.show(show_img_reordered)

    def draw_bounding_boxes(self) -> np.ndarray:
        """Utility function to draw bounding boxes on the image."""
        img_bb = self.image.copy()
        for (x1, y1, x2, y2), _ in self.bounding_boxes:
            cv2.rectangle(img_bb, (x1, y1), (x2, y2), (255, 0, 0), 1)
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
            logger.assertion(self._counterObj is not None, "Counter object not set.")
            self._bounding_boxes = cast(RxDetection, self._counterObj).count(self.image)
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


class RxImageSegment(RxImageCount):
    """Image class for segmentation methods. This class inherits from RxImageCount for
    the loading and counting methods.
    """

    def __init__(self) -> None:
        super().__init__()
        self._segmenterObj: Optional[RxSegmentation] = None
        self._background_mask: Optional[np.ndarray] = None
        self._masked_ROIs: Optional[List[np.ndarray]] = None
        self._detail_ROI_seg: Optional[List[List[SegmentResult]]] = None

    def set_segmenter(self, segmenterObj: RxSegmentation) -> None:
        """Sets the segmenter object. Reset any existing results to None when there's a new segmenter.

        Args:
            segmenterObj (Segmenter): Segmenter object.
        """
        self._segmenterObj = segmenterObj
        self._background_mask = None
        self._masked_ROIs = None
        self._detail_ROI_seg = None

    def visualize_background(self) -> None:
        """Visualize the background segment."""
        ts.show([self.image, self.background_segment])

    def visualize_masked_ROIs(self, img_per_row: int = 5) -> None:
        """
        Visualize the cropped segmentations.

        Args:
            img_per_row (int): The number of images per row.
        """
        show_imgs = self.masked_ROIs
        show_img_reordered = [
            show_imgs[i * img_per_row : i * img_per_row + img_per_row]
            for i in range(-(len(show_imgs) // -img_per_row))
        ]
        ts.show(show_img_reordered)

    @property
    def background_segment(self) -> np.ndarray:
        """
        Run full segmentation with SAM to separate forground/background.
        From the segmentation results, choose the one containing all pills.

        Returns:
            np.ndarray: The mask that separates the background and all the pills.
        """
        if self._background_mask is None:
            logger.assertion(self._segmenterObj is not None, "Segmenter object not set.")
            seg_mask_full = cast(RxSegmentation, self._segmenterObj).segment_full(self.image)
            self._background_mask = get_best_mask(seg_mask_full).astype(np.uint8)
        return self._background_mask

    @property
    def cropped_masks(self) -> List[np.ndarray]:
        """
        Return the cropped mask from the full background mask.
        # TODO: Replace the background (result of SAM) with instance segmentation.

        Returns:
            List[np.ndarray]: The cropped masks from the full background mask
        """

        bbox_xyxy_list = [item.bbox for item in self.bounding_boxes]
        cropped_masks = [self.background_segment[y1:y2, x1:x2] for (x1, y1, x2, y2) in bbox_xyxy_list]
        return cropped_masks

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
                [cv2.bitwise_or(ROI, ROI, mask=mask) for ROI, mask in zip(self.ROIs, self.cropped_masks)],
            )
        return self._masked_ROIs

    @property
    def ROI_segmentation(self) -> List[List[SegmentResult]]:
        """Check if the ROI segmentation has been produced. If not, segment before returning it.

        Returns:
            List[List[SegmentResult]]: Segmentation results for each ROI.
        """
        if self._detail_ROI_seg is None:
            logger.assertion(self._segmenterObj is not None, "Segmenter object not set.")
            self._detail_ROI_seg = [
                cast(RxSegmentation, self._segmenterObj).segment_ROI(ROI) for ROI in self.ROIs
            ]
        return self._detail_ROI_seg


class RxImageVerify(RxImageSegment):
    """Image class for verification methods. This class inherits from RxImageCount for the
    loading and counting methods.
    """

    def __init__(self) -> None:
        super().__init__()
        self._vectorized_ROIs: Optional[List[np.ndarray]] = None
        self._vectorized_ref: Optional[np.ndarray] = None
        self._similarity_scores: Optional[List[float]] = None
        self._vectorizerObj: Optional[RxVectorizer] = None

    def set_vectorizer(self, vectorizerObj: RxVectorizer) -> None:
        """Sets the vectorizer object. Reset any existing results to None when there's a new vectorizer.

        Args:
            vectorizerObj (vectorizer): Vectorizer object.
        """
        self._vectorizerObj = vectorizerObj
        self._vectorized_ROIs = None
        self._vectorized_ref = None
        self._similarity_scores = None

        # Default similarity function is cosine similarity
        self._similarity_fn = cosine_similarity

    def visualize_similarity_scores(self, img_per_col: int = 5) -> None:
        """Utility function to visualize similarity scores along with ROIs.

        Args:
            img_per_col (int): The number of images per column.
        """
        labels = [f"{i}: {score*100:.1f}%" for i, score in enumerate(self.similarity_scores)]
        self.visualize_ROIs(labels=labels)

    @property
    def vectorized_ref(self) -> np.ndarray:
        """Access property of the vectorized reference image; vectorize it if not already.

        Returns:
            np.ndarray: The vectorized reference image.
        """
        if self._vectorized_ref is None:
            logger.assertion(self._vectorizerObj is not None, "Vectorizer object not set.")
            self._vectorized_ref = cast(RxVectorizer, self._vectorizerObj).encode(self.ref_image)
        return self._vectorized_ref

    @property
    def vectorized_ROIs(self) -> List[np.ndarray]:
        """Access property of the vectorized ROIs; vectorize them if not already.

        Returns:
            List[np.ndarray]: The vectorized ROIs.
        """
        if self._vectorized_ROIs is None:
            logger.assertion(self._vectorizerObj is not None, "Vectorizer object not set.")
            self._vectorized_ROIs = cast(RxVectorizer, self._vectorizerObj).encode(self.masked_ROIs)
        return self._vectorized_ROIs

    @property
    def similarity_scores(self) -> List[float]:
        """Returns the similarity for all the ROIs."""
        if self._similarity_scores is None:
            self._similarity_scores = self._similarity_fn(
                [self.vectorized_ref], self.vectorized_ROIs
            ).squeeze()
        return self._similarity_scores


class RxImage(RxImageVerify):
    """Image class containing all the methods required for loading pill images, detection,
    segmentation, and verification.
    """

    def __init__(self) -> None:
        super().__init__()
