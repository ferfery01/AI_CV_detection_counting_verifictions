import pickle
from pathlib import Path
from typing import Any, Union

import yaml
from omegaconf import DictConfig, OmegaConf

from rx_connect.tools.logging import setup_logger

logger = setup_logger()

"""This module contains the functions for serialization and deserialization of objects.
"""


def read_pickle(path: Union[str, Path]) -> Any:
    """Reads a pickle file from disk and returns the deserialized object."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(data: Any, path: Union[str, Path]) -> None:
    """Writes a pickle file to disk from the given object."""
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved pickle file at {path}.")


def load_yaml(cfg: Union[str, Path]) -> DictConfig:
    """Loads the yaml file."""
    logger.assertion(Path(cfg).exists(), f"Couldn't find config file at {cfg}.")
    with open(cfg, "r") as f:
        data_loaded = yaml.safe_load(f)
    return OmegaConf.create(data_loaded)
