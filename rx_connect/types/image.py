import cv2
import numpy as np


class RxImageBase:
    def __init__(self):
        self._image_loaded = False

    def load_from_path(self, path):
        self._image = cv2.imread(path)
        self._image_loaded = True

    def load_from_camera(self):
        self._image = cv2.VideoCapture(0)
        self._image_loaded = True

    def load_ndarray(self, image: np.ndarray):
        self._image = image
        self._image_loaded = True

    def load_from_generator(self, generator_obj):
        raise NotImplementedError
        # self._image = generator.gen_image()
        # self._image_loaded = True

    def load_dummy_test(self):
        self._image = np.random.randint(256, size=(480, 640, 3)).astype(np.uint8)
        self._image_loaded = True

    def get_image(self):
        if not self._image_loaded:
            raise Exception("Image not loaded.")
        return self._image


class RxImageCount(RxImageBase):
    def __init__(self, inherit_image=None):
        super().__init__()
        self._bounding_boxes = None
        self._counterObj = None
        if isinstance(inherit_image, RxImageBase):
            self.__dict__.update(inherit_image.__dict__)

    def set_counter(self, counterObj):
        self._counterObj = counterObj

    def _count_pills(self):
        assert self._counterObj is not None, "Counter object not set."
        self._bounding_boxes = self._counterObj.count(self.get_image())

    def get_bounding_boxes(self):
        if self._bounding_boxes is None:
            self._count_pills()
        return self._bounding_boxes
        # return format: [((X, Y, W, H), score), ...]

    def get_count(self):
        return len(self.get_bounding_boxes())

    def get_ROI(self):
        return [self._image[x : (x + w), y : (y + h)] for (x, y, w, h), _ in self.get_bounding_boxes()]


class RxImageCountSegment(RxImageCount):
    def __init__(self, inherit_image=None):
        super().__init__()
        self._seg_mask_full = None
        self._seg_mask_ROI = None
        self._segmenterObj = None
        if isinstance(inherit_image, RxImageBase):
            self.__dict__.update(inherit_image.__dict__)

    def set_segmenter(self, segmenterObj):
        self._segmenterObj = segmenterObj

    def _segment_full(self):
        assert self._segmenterObj is not None, "Segmenter object not set."
        self._seg_mask_full = self._segmenterObj.segment(self.get_image())

    def _segment_ROI(self):
        assert self._segmenterObj is not None, "Segmenter object not set."
        self._seg_mask_ROI = [self._segmenterObj.segment(ROI) for ROI in self.get_ROI()]

    def get_full_segmentation(self):
        if self._seg_mask_full is None:
            self._segment_full()
        return self._seg_mask_full

    def get_ROI_segmentation(self):
        if self._seg_mask_ROI is None:
            self._segment_ROI()
        return self._seg_mask_ROI
