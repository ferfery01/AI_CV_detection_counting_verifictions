from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

import numpy as np

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

    def __init__(self, cfg: Union[str, Path]) -> None:
        self._cfg = cfg
        self._load_cfg()
        self._load_model()

    @abstractmethod
    def _load_cfg(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _load_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _predict(self, image: np.ndarray) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _postprocess(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
