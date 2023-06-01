from typing import Any, Union


def safe_cast(input: Union[None, Any]) -> Any:
    if input is None:
        raise TypeError("input cannot be None")
    return input
