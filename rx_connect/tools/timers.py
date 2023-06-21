import time
from functools import wraps
from typing import Any, Callable

from rx_connect.tools.logging import setup_logger

logger = setup_logger()


def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Timer decorator.
    Usage example:
        @timer
        def your_function():
            pass

    """

    @wraps(func)
    def timed_func(*args, **kwargs):
        begin = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"Function '{func.__name__}' took {end - begin:.5} seconds.")
        return result

    return timed_func
