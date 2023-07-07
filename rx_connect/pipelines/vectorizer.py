from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, List, Optional, Union, cast

import cv2
import numpy as np
from vlad import VLAD

from rx_connect import CACHE_DIR, SHARED_REMOTE_DIR
from rx_connect.core.utils.cv_utils import equalize_histogram
from rx_connect.tools.data_tools import fetch_from_remote
from rx_connect.tools.logging import setup_logger
from rx_connect.tools.serialization import read_pickle
from rx_connect.tools.timers import timer

logger = setup_logger()


class RxVectorizer(ABC):
    def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
        """Initializes the RxVectorizer object.

        Args:
            model_path (str): Path to the vectorizer model.

        Raises:
            AssertionError: If the model path does not exist.
        """
        # If remote model path is provided, fetch the model from the remote

        if model_path is not None:
            self._model_path = fetch_from_remote(model_path, cache_dir=CACHE_DIR / "verification")
            logger.assertion(self._model_path.exists(), f"Model path {self._model_path} does not exist.")
            self._load_model()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Inference call."""
        image_preproc = self._preprocess(image)
        return self._predict(image_preproc)

    @timer
    def encode(self, images: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Encodes the image using the VLAD model. This is an alias for __call__."""
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                return self(images)
            logger.assertion(
                images.ndim == 4, "Images must be 3-dimensional (single) or 4-dimensional (stacked)."
            )
        return [self(image) for image in images]

    @abstractmethod
    def _load_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _preprocess(self, image: np.ndarray) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _predict(self, image_preproc: Any) -> Any:
        raise NotImplementedError


class RxVectorizerSift(RxVectorizer):
    num_features: ClassVar[int] = 1000

    def __init__(
        self,
        model_path: Union[str, Path] = f"{SHARED_REMOTE_DIR}/checkpoints/verification/vlad_1000_rn_16_v1.pkl",
    ) -> None:
        """Initializes the RxVectorizerSift object.

        Args:
            model_path (str): Path to the VLAD model.

        Raises:
            AssertionError: If the model path does not exist.
        """
        super().__init__(model_path)

    def _load_model(self) -> None:
        """Loads the VLAD model from the given path."""
        self._model = cast(VLAD, read_pickle(self._model_path))
        self._model.verbose = False
        self._SIFT = cv2.SIFT_create(self.num_features)  # type: ignore

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Extracts the image descriptors using SIFT. The histogram of the colored image is
        equalized before extracting the SIFT descriptors.
        """
        logger.assertion(image.ndim == 3, f"Image should be a 3D array, but got a {image.ndim}D array.")

        # Equalize the histogram of the image
        image = equalize_histogram(image)

        # Extract the SIFT image descriptor
        _, descs = self._SIFT.detectAndCompute(image, None)
        desc = descs[: self.num_features]

        return desc

    def _predict(self, descriptor: np.ndarray) -> np.ndarray:
        """Encodes the image descriptor using the VLAD model.

        The input descriptor should be a 2D array with shape (num_features, 128).
        The output vector will be a 2D array with shape (1, num_clusters * 128).
        """
        return self._model.transform([descriptor])


class RxVectorizerColorhist(RxVectorizer):
    def _load_model(self) -> None:
        pass

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        return image

    def _predict(self, image: np.ndarray) -> np.ndarray:
        """Encodes the image using the colorhist technique."""

        # Compute image vector
        # image pixel values array
        # which channels do you (all RGB channels here)
        # how many descriptors do you want per channel?
        # what is the range of values per color channel?
        hist = cv2.calcHist(
            [image],
            [0, 1, 2],
            None,  # mask, not used in this
            [256, 256, 256],
            [0, 256, 0, 256, 0, 256],
        )
        final_vector = cv2.normalize(hist, hist).flatten()

        return final_vector
