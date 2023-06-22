import sys
from pathlib import Path
from typing import Union

from rx_connect import SHARED_REMOTE_DIR


def is_gpu_server() -> bool:
    """Return True if the current machine is a GPU server else False.

    NOTE: It is assumed that the GPU server is running Ubuntu.
    """
    return True if sys.platform == "linux" else False


def is_remote_dir(dir_path: Union[str, Path]) -> bool:
    """Return True if `dir_path` is a remote directory but the current machine is not a
    GPU server else False.
    """
    if str(dir_path).startswith(SHARED_REMOTE_DIR) and not is_gpu_server():
        return True
    else:
        return False
