from typing import Iterable, Optional, Union

import torch


def clip_grads_fn(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    max_norm: Optional[float] = None,
    norm_type: float = 2,
    error_if_nonfinite: bool = True,
) -> Optional[torch.Tensor]:
    """Perform gradient clipping for the optimizer parameters. Called before optimizer step.
    The norm is computed over all gradients together, as if they were concatenated into a single
    vector. Gradients are modified in-place.

    Parameters:
        parameters (Tensor): The gradients to be clipped.
        max_norm (float, optional): The maximum norm at which to clip the gradients. If `max_norm > 0`,
            the gradients will be clipped in order to prevent the norm to exceed `max_norm`. If `None`,
            no clipping is performed.
        norm_type (float): The gradient clipping algorithm to use. Can be `inf` for infinity norm.
        error_if_nonfinite (bool): If True, an error is thrown if the total norm of the gradients
            from `parameters` is `nan`, `inf`, or `-inf`.

    Returns:
        Optional[Tensor]: The clipped gradients. If max_norm is <= 0, then the gradients are
            not clipped.
    """
    if max_norm is not None and max_norm >= 0:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite)
    else:
        return None
