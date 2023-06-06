from typing import List, Tuple

import numpy as np
import torch

from rx_connect.tools.data_tools import fetch_from_remote


class Counter:
    """
    Counter inference class. This class implements the functions to load model and predict.
    """

    def __init__(self, model_path: str = "remote/path/to/model") -> None:
        """
        Constructor.

        Args:
            model_path (str, optional): Path to the model. Defaults to "remote/path/to/model".
        """
        self._model_path = model_path
        self._load_model()

    def _load_model(self):
        """
        Loads the model.
        """
        self._model = torch.load(fetch_from_remote(self._model_path))

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        return image

    def _predict(self, image: np.ndarray) -> np.ndarray:
        """
        Detection network inference function.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray:
                List of bounding boxes.
                Expects shape of N x 5, N is the number of detected bounding boxes.
                Each bounding box is (X, Y, W, H, S) where X, Y, W, H are the coordinates
                of the bounding box and S is the score.
        """
        return self._model.predict(image)

    def count(self, image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Count function that calls the prediction and format the result.

        Args:
            image (np.ndarray): Input image. Single image is expected.

        Returns:
            List[Tuple[np.ndarray, float]]: List of (bounding box and score) pairs.
        """
        assert len(image.shape) == 3, "Single image is expected."
        image_preproc = self._preprocess(image=image)
        results = self._predict(image=image_preproc)

        return [(bbox[:4], bbox[4]) for bbox in results]
