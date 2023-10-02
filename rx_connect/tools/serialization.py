import pickle
from pathlib import Path
from typing import Any, Union, cast

from omegaconf import DictConfig, OmegaConf

from rx_connect.tools.logging import setup_logger

logger = setup_logger()

"""This module contains the functions for serialization and deserialization of objects.
"""


def read_pickle(path: Union[str, Path]) -> Any:
    """Reads a pickle file from disk and returns the deserialized object."""
    assert Path(path).exists(), f"Couldn't find pickle file at {path}."
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(data: Any, path: Union[str, Path]) -> None:
    """Writes a pickle file to disk from the given object."""
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved pickle file at {path}.")


def load_yaml(cfg: Union[str, Path]) -> DictConfig:
    """Load and resolve a YAML configuration file into a DictConfig object.

    This function loads a YAML configuration file, resolves any variable interpolations
    present, and returns a DictConfig object containing the fully-resolved configuration.

    Args:
        cfg (Union[str, Path]): The path to the YAML configuration file to load.
            This can be either a string or a Path object.

    Returns:
        DictConfig: A DictConfig object containing the resolved configuration data.

    Raises:
        OmegaConfException: If there is an error in the YAML file or in resolving interpolations.
    """
    # Load the YAML file into an OmegaConf object
    data_loaded = OmegaConf.load(cfg)

    # Convert the DictConfig to a regular dict while resolving interpolations
    resolved_conf = OmegaConf.to_container(data_loaded, resolve=True)

    # Convert the resolved dictionary back to a DictConfig
    return cast(DictConfig, OmegaConf.create(resolved_conf))
