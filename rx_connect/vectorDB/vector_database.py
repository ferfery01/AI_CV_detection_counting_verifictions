from pathlib import Path
from typing import Union

import cv2
import numpy as np
from sklearn.cluster import KMeans

from rx_connect import SHARED_EPILL_DATA_DIR
from rx_connect.generator.io_utils import (
    load_pill_mask_paths,
    load_random_pills_and_masks,
)
from rx_connect.pipelines.vectorizer import RxVectorizer
from rx_connect.tools.logging import setup_logger
from rx_connect.tools.serialization import write_pickle

logger = setup_logger()


class VectorDB:
    def __init__(
        self,
        vectorizer: RxVectorizer,
        image_dir: Union[str, Path] = SHARED_EPILL_DATA_DIR,
        num_image_samples: int = 1024,
        num_dim: int = 128,
    ) -> None:
        self._vectorizer = vectorizer
        self._image_dir = image_dir
        self._num_image_samples = num_image_samples
        self._load_masked_images()
        self._gen_vectorSpace(num_dim)

    def _load_masked_images(self) -> None:
        """
        Load the paths for images and masks. Randomly sample and load them. Apply mask to images and return.
        """
        self._image_mask_paths = load_pill_mask_paths(self._image_dir)
        pill_images, pill_masks = load_random_pills_and_masks(
            self._image_mask_paths, pill_types=self._num_image_samples
        )
        self._masked_images = [
            cv2.bitwise_or(image, image, mask=mask) for image, mask in zip(pill_images, pill_masks)
        ]

    def _gen_vectorSpace(self, num_dim: int = 20) -> None:
        """
        Vectorize the sampled images. Cluster the vectors to generate the vector space.

        Args:
            num_dim: The number of dimensions of the output vector space.
        """
        assert num_dim <= len(
            self._masked_images
        ), f"Number of samples {self._num_image_samples} is less than clustering dimensions {num_dim}."
        all_vectors = np.stack(self._vectorizer.encode(self._masked_images), axis=0)
        self._vectorSpace = (
            KMeans(n_clusters=num_dim, random_state=0, n_init="auto").fit(all_vectors).cluster_centers_
        )

    def export(self, filepath: Union[str, Path]) -> None:
        """
        Export the vectorizer type and the vector space as a pickle file.

        Args:
            filepath: The path to save the pickle file.
        """
        filepath = Path(filepath)
        dict_to_export = {
            "vectorizer": type(self._vectorizer),
            "vectorSpace": self._vectorSpace,
        }
        write_pickle(dict_to_export, filepath)
