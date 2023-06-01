from typing import List, Optional, Tuple

import cv2
import numpy as np

from rx_connect.tools.casting import safe_cast


class RxImageBase:
    """
    Base class for RxImage objects. This class implements the loading methods.
    """

    def __init__(self):
        self._image = None

    def load_from_path(self, path: str):
        """
        Loads the image from the given path.

        Args:
            path (str): Path to the image file.
        """
        self._image = cv2.imread(path)

    def load_from_camera(self):
        """
        Loads the image from default camera.
        """
        self._image = cv2.VideoCapture(0)

    def load_ndarray(self, image: np.ndarray):
        """
        Loads the image from the given numpy array.

        Args:
            image (np.ndarray): Image as numpy array.
        """
        self._image = image

    def load_from_generator(self, generator_obj):
        """
        Loads the image from the given generator object.

        Args:
            generator_obj (Generator): Generator object.
        """
        raise NotImplementedError
        # self._image = generator.gen_image()

    def load_dummy_test(self):
        """
        Dummy load funciton to generate random test images.
        """
        self._image = np.random.randint(256, size=(480, 640, 3)).astype(np.uint8)

    def get_image(self) -> np.ndarray:
        """
        Check if the image has been loaded, and return the image.

        Returns:
            np.ndarray: Loaded image.
        """
        assert self._image is not None, "Image not loaded."
        return self._image


class RxImageCount(RxImageBase):
    """
    Image class for count counting methods. This class inherits from RxImageBase for the loading methods.
    """

    def __init__(self, inherit_image: Optional[RxImageBase] = None):
        """
        Constructor.

        Args:
            inherit_image (Optional[RxImageBase]): Inherited image.
        """
        super().__init__()
        self._bounding_boxes = None
        self._counterObj = None
        if isinstance(inherit_image, RxImageBase):
            self.__dict__.update(inherit_image.__dict__)

    def set_counter(self, counterObj):
        """
        Sets the counter object.

        Args:
            counterObj (Counter): Counter object.
        """
        self._counterObj = counterObj

    def _count_pills(self):
        """
        Check if the counter object has been set, and count the number of pills in the image.
        """
        assert self._counterObj is not None, "Counter object not set."
        self._bounding_boxes = self._counterObj.count(self.get_image())

    def get_bounding_boxes(self) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Check if the bounding boxes has been produced. If not, count before returning them.

        Returns:
            List[Tuple[Tuple[int, int, int, int], float]]: Bounding boxes. Format: [((X, Y, W, H), score), ...]
        """
        if self._bounding_boxes is None:
            self._count_pills()
        return safe_cast(self._bounding_boxes)

    def get_count(self) -> int:
        """
        Return the length of the bounding boxes as the count.

        Returns:
            int: Count.
        """
        return len(self.get_bounding_boxes())

    def get_ROI(self) -> List[np.ndarray]:
        """
        Return the cropped image views as the regions of interest (ROI).

        Returns:
            List[np.ndarray]: List of cropped image views.
        """
        return [self._image[x : (x + w), y : (y + h)] for (x, y, w, h), _ in self.get_bounding_boxes()]


class RxImageCountSegment(RxImageCount):
    """
    Image class for segmentation methods. This class inherits from RxImageCount for the loading and counting methods.
    """

    def __init__(self, inherit_image=None):
        """
        Constructor.

        Args:
            inherit_image (Optional[RxImageBase]): Inherited image.
        """
        super().__init__()
        self._seg_mask_full = None
        self._seg_mask_ROI = None
        self._segmenterObj = None
        if isinstance(inherit_image, RxImageBase):
            self.__dict__.update(inherit_image.__dict__)

    def set_segmenter(self, segmenterObj):
        """
        Sets the segmenter object.

        Args:
            segmenterObj (Segmenter): Segmenter object.
        """
        self._segmenterObj = segmenterObj

    def _segment_full(self):
        """
        Check if the segmenter object has been set, and segment the full image.
        """
        assert self._segmenterObj is not None, "Segmenter object not set."
        self._seg_mask_full = self._segmenterObj.segment(self.get_image())

    def _segment_ROI(self):
        """
        Check if the segmenter object has been set, and segment the ROI.
        """
        assert self._segmenterObj is not None, "Segmenter object not set."
        self._seg_mask_ROI = [self._segmenterObj.segment(ROI) for ROI in self.get_ROI()]

    def get_full_segmentation(self) -> np.ndarray:
        """
        Check if the full segmentation has been produced. If not, segment before returning it.

        Returns:
            np.ndarray: Full segmentation.
        """
        if self._seg_mask_full is None:
            self._segment_full()
        return self._seg_mask_full

    def get_ROI_segmentation(self) -> List[np.ndarray]:
        """
        Check if the ROI segmentation has been produced. If not, segment before returning it.

        Returns:
            List[np.ndarray]: List of ROI segmentations.
        """
        if self._seg_mask_ROI is None:
            self._segment_ROI()
        return safe_cast(self._seg_mask_ROI)
