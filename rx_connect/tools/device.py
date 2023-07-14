import re
from typing import List, Optional, Union

import torch


def find_free_gpu() -> Optional[torch.device]:
    """Iteratively searches for a GPU that currently has no memory allocated on it.
    As soon as a free GPU is found, it is returned. If no free GPU is found, None is returned.
    """
    # Get the number of available GPUs
    gpu_count = torch.cuda.device_count()

    # Iterate over each GPU
    for i in range(gpu_count):
        # Create the GPU device string
        device_str = f"cuda:{i}"

        # Check the memory allocated on the current GPU
        allocated = torch.cuda.memory_allocated(device_str)

        # If no memory is allocated, return this GPU as it is free
        if allocated == 0:
            return torch.device(device_str)

    # If no free GPU was found, return None
    return None


def get_best_available_device(allow_mps: bool = False) -> torch.device:
    """Finds the best available device in the order CUDA -> MPS -> CPU.

    Parameters:
        allow_mps (bool): Whether to all the function to return MPS device if available

    Returns:
        torch.device: The available device.
    """
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Find a free GPU
        device = find_free_gpu()
        if device is not None:
            return device

    # If no free CUDA device is available, check if MPS is available
    if allow_mps and torch.backends.mps.is_available():
        return torch.device("mps")

    # If no free CUDA device is available and MPS is not available, return CPU
    return torch.device("cpu")


def check_format(s: str) -> bool:
    """Check if a string is a comma-separated list of integers."""
    pattern = r"^\d+(,\s*\d+)*$"
    return bool(re.match(pattern, s))


def parse_cuda_for_devices(cuda: Optional[str] = None) -> Union[int, List[int]]:
    """Parse the cuda flag to return a list of device ids.

    - If cuda is None, return 1.
    - If cuda is -1, it returns all the available CUDA devices.
    - If cuda is a device id, it returns [device id].
    - If cuda is "1, 3", it returns [1, 3].

    Raise an error if the cuda flag is malformed.
    """
    if cuda is None:
        return 1

    if cuda is not None and not torch.cuda.is_available():
        raise ValueError("CUDA is not available.")

    if not check_format(cuda):
        raise ValueError("CUDA flag is malformed.")

    devices: List[int] = [int(x) for x in cuda.strip().split(",") if int(x) < torch.cuda.device_count()]
    if len(devices) == 1 and devices[0] == -1:
        return list(range(torch.cuda.device_count()))
    else:
        return devices
