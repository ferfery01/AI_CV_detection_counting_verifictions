from pathlib import Path
from typing import ClassVar, Union, cast

import cv2
import numpy as np
from vlad import VLAD

from rx_connect import SHARED_REMOTE_DIR
from rx_connect.core.utils.cv_utils import equalize_histogram
from rx_connect.tools.data_tools import fetch_from_remote
from rx_connect.tools.logging import setup_logger
from rx_connect.tools.serialization import read_pickle

logger = setup_logger()


class RxVectorization:
    num_features: ClassVar[int] = 1000

    def __init__(
        self, model_path: Union[str, Path] = "RxConnectShared/checkpoints/verification/vlad_1000_rn_16_v1.pkl"
    ) -> None:
        """Initializes the RxVerification object.

        Args:
            model_path (str): Path to the VLAD model.

        Raises:
            AssertionError: If the model path does not exist.
        """
        # If remote model path is provided, fetch the model from the remote
        if str(model_path).startswith(SHARED_REMOTE_DIR):
            model_path = fetch_from_remote(model_path, cache_dir=".cache/verification")

        logger.assertion(Path(model_path).exists(), f"Model path {model_path} does not exist.")
        self._model_path = model_path

        # Load the VLAD model and set the verbosity to False
        self._model = self._load_model()
        self._model.verbose = False

        self._SIFT = cv2.SIFT_create(self.num_features)

    def _load_model(self) -> VLAD:
        """Loads the VLAD model from the given path."""
        model = read_pickle(self._model_path)
        return cast(VLAD, model)

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

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Encodes the image using the VLAD model."""

        # Compute image descriptors
        image_desc = self._preprocess(image)

        # Extract the final vector
        final_vector = self._predict(image_desc)

        return final_vector

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encodes the image using the VLAD model. This is an alias for __call__."""
        return self(image)
