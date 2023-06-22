from typing import Optional

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
