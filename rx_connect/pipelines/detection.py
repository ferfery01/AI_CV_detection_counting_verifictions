from pathlib import Path
from typing import List, Union

import numpy as np
from super_gradients.training import models as yolo_model
from super_gradients.training.models.predictions import DetectionPrediction

from rx_connect import CACHE_DIR, PIPELINES_DIR
from rx_connect.core.types.detection import CounterModuleOutput
from rx_connect.pipelines.base import RxBase
from rx_connect.tools.data_tools import fetch_from_remote
from rx_connect.tools.logging import setup_logger
from rx_connect.tools.serialization import load_yaml
from rx_connect.tools.timers import timer

logger = setup_logger()


class RxDetection(RxBase):
    def __init__(self, cfg: Union[str, Path] = f"{PIPELINES_DIR}/configs/Dev/counter_config.yml") -> None:
        super().__init__(cfg)

    def _load_cfg(self) -> None:
        """Loads the config file and sets the attributes."""
        conf = load_yaml(self._cfg)

        self._model_path = conf.detection.model_path
        self._yolo_model = conf.detection.model
        self._conf = conf.detection.conf
        self._n_classes = len(conf.detection.classes)

    def _load_model(self) -> None:
        """Loads the YOLO-NAS model."""
        # Fetch the model from the remote if it is not already in the cache.
        self._model_path = fetch_from_remote(self._model_path, cache_dir=CACHE_DIR / "counting")
        logger.assertion(self._model_path.exists(), f"Model path {self._model_path} does not exist.")

        # Load the model.
        self._model = yolo_model.get(
            self._yolo_model, num_classes=self._n_classes, checkpoint_path=str(self._model_path)
        )
        self._model.eval()

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        return image

    def _predict(self, image: np.ndarray) -> DetectionPrediction:
        """Detection network inference function. This function takes an image and returns a list
        of dictionaries with keys: bboxes_xyxy, confidence, class_id, class_name, and class_confidence.
        """
        model_pred = self._model.predict(image, self._conf)
        return list(model_pred._images_prediction_lst)[0].prediction

    def _postprocess(self, model_pred: DetectionPrediction) -> DetectionPrediction:
        return model_pred

    def __call__(self, image: np.ndarray) -> DetectionPrediction:
        image = self._preprocess(image)
        results = self._predict(image)
        output_results = self._postprocess(results)

        return output_results

    @timer()
    def count(self, image: np.ndarray) -> List[CounterModuleOutput]:
        """Counts the number of objects in the image and returns a list of CounterModuleOutput objects."""
        logger.assertion(image.ndim == 3, f"Image should be a 3D array, but got a {image.ndim}D array.")

        pred = self(image)
        bboxes: List[List[int]] = pred.bboxes_xyxy.astype(int).tolist()
        scores: List[float] = pred.confidence.tolist()

        return [CounterModuleOutput(bbox, score) for bbox, score in zip(bboxes, scores)]


if __name__ == "__main__":
    from rx_connect import SHARED_EPILL_DATA_DIR
    from rx_connect.pipelines.image import RxImage

    img_path = f"{SHARED_EPILL_DATA_DIR}/images/68258-6044_0_1.jpg"

    # Test the class
    image_obj = RxImage()
    image_obj.load_image(img_path)

    pill_count_obj = RxDetection()
    image_obj.set_counter(pill_count_obj)

    logger.info(f"Number of pills: {image_obj.pill_count}.")
