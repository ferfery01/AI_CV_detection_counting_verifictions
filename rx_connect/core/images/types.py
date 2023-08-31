from typing import Dict, Optional

import numpy as np
import skimage as ski
import torch

# Dictionary mapping python built-in types and numpy dtypes to torch dtypes
TORCH_DTYPE: Dict[type, torch.dtype] = {
    float: torch.float32,
    np.double: torch.double,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.int32: torch.int32,
    int: torch.int64,
    np.int64: torch.int64,
    np.uint8: torch.uint8,
    bool: torch.bool,
    np.bool_: torch.bool,
    np.bool8: torch.bool,
}


def img_to_tensor(image: np.ndarray, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Converts numpy image (RGB, BGR, Grayscale, Mask) to a `torch.Tensor`. The numpy `HWC`
    image is converted to `CHW` tensor. If the image is in `HW` format (grayscale, mask), it will
    be converted to pytorch `HW` tensor. This function converts the image to the specified dtype
    and properly rescales the values.

    Args:
        image (np.ndarray): The image to convert.
        dtype (Optional[torch.dtype], optional): The dtype to convert the image to. If None, the
            dtype of the image is used.

    Returns:
        torch.Tensor of shape (C, H, W) or (H, W).
    """
    # Check dtype argument
    if dtype is not None and not isinstance(dtype, torch.dtype):
        raise TypeError(f"dtype must be a torch.dtype, got {type(dtype)}")

    # Check image dimensions
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must have shape (H, W) or (H, W, C), got {image.shape}")

    # If 2D image, expand dimensions to make it a single channel image
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    # Transpose to make the array channel-first
    image = image.transpose((2, 0, 1))

    # Determine the dtype for the output tensor. If not specified, look up the PyTorch dtype
    # that corresponds to the numpy dtype of a sample element (image[0, 0, 0]). If it's not
    # in TORCH_DTYPE, use the original dtype.
    dtype = dtype or TORCH_DTYPE.get(type(image[0, 0, 0]), image.dtype)
    if dtype not in TORCH_DTYPE.values():
        raise ValueError(f"Unsupported dtype {dtype}")

    # Conversion based on dtype
    if dtype == torch.bool:
        image = ski.img_as_bool(image)
    elif dtype == torch.uint8:
        image = ski.img_as_ubyte(image)
    elif dtype == torch.int16:
        image = ski.img_as_int(image)
    elif dtype in (torch.float, torch.float16, torch.float32, torch.float64):
        image = ski.img_as_float(image)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    # Convert numpy array to torch tensor
    return torch.tensor(image, dtype=dtype)
