import functools
import inspect
import time
from datetime import timedelta
from logging import Logger
from types import FrameType
from typing import Any, Callable, Optional, TypeVar, cast

from rx_connect.tools.logging import setup_logger

logger = setup_logger()

F = TypeVar("F", bound=Callable[..., Any])
"""TypeVar for a generic function.
"""


class timer:
    """Context manager and decorator for timing a block of code or function.

    This class can be used to measure execution time of a specific block of code or function.
    The measured time is accessible via the `duration` property, and can be automatically logged
    by providing a `logger` object during initialization.

    Example use case - Timing a function `f`:
    >>> def f():
    >>>     time.sleep(1) # simulate "long computation"

    Directly use the timer and grab `duration` after the context block has finished:
    >>> with timer() as block_timer:
    >>>     f()
    >>> print(f"{block_timer.duration:0.4f}s")

    Alternatively, you can use timer as a decorator:

    >>> @timer(logger=logging.getLogger(__name__))
    >>> def f():
    >>>     time.sleep(1)  # This function execution time will be logged

    Another use case involves passing a `name` and a `logger`. The timing will be recorded
    when the context block is exited:
    >>> log = logging.getLogger("my-main-program")
    >>> with timer(logger=log, name="timing-func-f"):
    >>>     f()

    NOTE: that the duration is in seconds. It is a `float`, so it can represent fractional seconds.
    """

    __slots__ = ("logger", "name", "_duration", "start")

    def __init__(self, logger: Optional[Logger] = None, name: str = "Code Block") -> None:
        if logger is not None and not isinstance(logger, Logger):
            raise TypeError(f"logger must be of type logging.Logger or CustomLogger, got {type(logger)}")

        self.logger = logger
        self.name = name
        self._duration: Optional[float] = None

        self.start: float = -1.0
        """for start, -1 is the uninitialized value.
        It is set at the context-block entering method: __enter__
        """

    def __enter__(self) -> "timer":
        """Start timing when entering the context block.

        This function records the monotonic time at the start of the context block or function
        to ensure the subsequent code execution does not affect the start time.
        """
        self.start = time.monotonic()
        return self

    def __exit__(self, *args) -> None:
        """End timing when exiting the context block.

        This function calculates the time duration of the context block or function execution.
        """
        # calculate the duration *first*
        self._duration = time.monotonic() - self.start

        # i.e. validation occurs after
        if self.start == -1:
            raise ValueError(
                "Cannot use context-block exit method if context-block enter method has not been called!"
            )
        self.log_end_time()

    def __call__(self, func: F) -> F:
        """Allows the timer class to be used as a decorator.

        When the timer instance is called as a decorator, it creates a new timer instance with the name
        set to the name of the function being decorated. It then wraps the function in a context block
        such that the function's execution time is recorded.

        Args:
            func (F): The function to be timed.

        Returns:
            F: The wrapped function, which when called will have its execution time recorded.
        """
        # create a timer with logger and name set to func.__name__
        timer_instance = self.__class__(logger=logger, name=f"`{func.__name__}`")

        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with timer_instance:
                return func(*args, **kwargs)

        return cast(F, decorate_context)

    def log_end_time(self) -> None:
        if self.logger is not None:
            caller_namespace = "<unknown_caller_namespace>"
            frame: Optional[FrameType] = inspect.currentframe()
            if frame is not None:
                frame = frame.f_back
                if frame is not None:
                    caller_namespace = frame.f_globals["__name__"]
            metric_name = caller_namespace
            if self.name:
                metric_name = f"{metric_name}.{self.name}"
            self.logger.info(f"{self.name if self.name else metric_name} took {self._duration:5.2f}s")

    @property
    def duration(self) -> float:
        """Retrieve the execution time duration in seconds.

        The property represents the number of seconds from when the context block was entered until it was exited.
        If the context block was either not entered, or entered but not exited, a ValueError is raised.
        """
        if self._duration is None:
            raise ValueError("Cannot get duration if timer has not exited context block!")
        else:
            return self._duration

    @property
    def timedelta(self) -> timedelta:
        """Return the execution time duration as a datetime timedelta object.

        The method formats the `duration` as a datetime timedelta object for convenient
        representation and manipulation of time duration.
        """
        return timedelta(seconds=self.duration)

    def __format__(self, format_spec: str) -> str:
        """String representation of the timer's duration, using the supplied formatting specification."""
        return f"{self.duration:{format_spec}}"

    def __float__(self) -> float:
        """Alias to the timer's duration. See :func:`duration` for specification."""
        return self.duration

    def __int__(self) -> int:
        """Rounds the duration to the nearest second."""
        return int(round(float(self)))

    # Implementations for builtins numeric operations.

    def __eq__(self, other) -> bool:
        return float(self) == float(other)

    def __lt__(self, other) -> bool:
        return float(self) < float(other)

    def __le__(self, other) -> bool:
        return float(self) <= float(other)

    def __gt__(self, other) -> bool:
        return float(self) > float(other)

    def __ge__(self, other) -> bool:
        return float(self) >= float(other)

    def __abs__(self) -> float:
        return abs(float(self))

    def __add__(self, other) -> float:
        return float(self) + float(other)

    def __sub__(self, other) -> float:
        return float(self) - float(other)

    def __mul__(self, other) -> float:
        return float(self) * float(other)

    def __floordiv__(self, other) -> int:
        return int(self) // int(other)

    def __truediv__(self, other) -> float:
        return float(self) / float(other)
