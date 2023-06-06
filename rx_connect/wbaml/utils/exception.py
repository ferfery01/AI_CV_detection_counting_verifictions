from typing import Sequence

from rx_connect.wbaml.utils.logging import setup_logger

__all__: Sequence[str] = ("assert_condition",)

logger = setup_logger()


def assert_condition(condition: bool, message: str, stacklevel: int = 2) -> None:
    """Assert a condition and log the exception if it fails.

    Usage:
        >>> assert_condition(x == 0, f"expected x to be 0, got {x}")
        >>> [06/03/23 16:31:58] ERROR [rx_connect/wbaml/utils/exception.py] [assert_condition: 27] expected x to be 0,
              got 1
    ╭────────────────────────────────────────────────────── Traceback (most recent call last) ─────────────────────────╮
    │ /Users/dvsingxe/ai-lab-RxConnect/rx_connect/pill_validation/test.py:19 in assert_condition                       │
    │                                                                                                                  │
    │   16                                           ╭──────────────────────── locals ─────────────────────────╮       │
    │   17 def assert_condition(condition, message): │ condition = False                                       │       │
    │   18 │   try:                                  │         e = AssertionError('expected x to be 0, got 1') │       │
    │ ❱ 19 │   │   assert condition, message         │   message = 'expected x to be 0, got 1'                 │       │
    │   20 │   except AssertionError as e:           ╰─────────────────────────────────────────────────────────╯       │
    │   21 │   │   logger.exception("oopsie")                                                                          │
    │   22 │   │   raise                                                                                               │
    ╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
    """
    try:
        assert condition, message
    except AssertionError as e:
        logger.exception(message, stack_info=True, exc_info=True, stacklevel=stacklevel)
        raise e
