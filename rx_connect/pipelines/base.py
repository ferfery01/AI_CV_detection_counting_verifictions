from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from rx_connect.tools.device import get_best_available_device
from rx_connect.tools.logging import setup_logger

logger = setup_logger()


class RxBase(ABC):
    """This is the abstract class for all the RxConnect objects. This class cannot be instantiated
    and is supposed to be inherited by all the RxConnect objects.

    It requires the implementation of the following methods:
        - _load_cfg
        - _load_model
        - _preprocess
        - _predict
        - _postprocess
    """

    def __init__(self, cfg: Union[str, Path], device: Optional[Union[str, torch.device]]) -> None:
        self._cfg = cfg
        self._device = device or get_best_available_device()
        self._load_cfg()
        self._load_model()

    @abstractmethod
    def _load_cfg(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _load_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _preprocess(self, image: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _predict(self, image: Union[np.ndarray, torch.Tensor]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _postprocess(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __call__(self, image: np.ndarray) -> Any:
        """This method is the main entry point of the RxConnect objects. It takes an image and returns
        the prediction.
        """
        processed_image = self._preprocess(image)
        prediction = self._predict(processed_image)
        processed_result = self._postprocess(prediction)
        return processed_result
