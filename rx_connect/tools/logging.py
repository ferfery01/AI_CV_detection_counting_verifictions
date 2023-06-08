import logging
from logging import LogRecord
from pathlib import Path
from typing import Sequence

from rich.logging import RichHandler
from rich.text import Text

__all__: Sequence[str] = ("setup_logger",)


_PACKAGE_NAME: str = "rx_connect"


class CustomRichHandler(RichHandler):
    """Custom handler based on RichHandler with additional functionality."""

    def get_relative_path(self, path_name: str) -> str:
        """Get the relative path from the root directory to the parent directory."""
        _path = Path(path_name)

        # Find the root directory of the module
        module_root_dir = None
        for p in reversed(_path.parents):
            if p.name == _PACKAGE_NAME:
                module_root_dir = p
                break

        if module_root_dir is not None:
            # Get the relative path from the root directory to the parent directory
            relative_path = _path.relative_to(module_root_dir.parent)
            return str(relative_path)
        else:
            return str(_path)

    def render_message(self, record: LogRecord, message: str) -> Text:
        """Render the log message with custom formatting.

        Args:
            record (LogRecord): The log record.
            message (str): The log message.

        Returns:
            The rendered log message as a Rich Text object.
        """
        relative_path = self.get_relative_path(record.pathname)

        text = Text()
        text.append(f"[{relative_path}]", style="light_cyan1")
        text.append(f" [{record.funcName}: {record.lineno}]", style="thistle1")
        text.append(f" {record.msg}")

        return text


class CustomLogger(logging.RootLogger):
    def __init__(self, level: int = logging.INFO) -> None:
        super().__init__(level)

    def assertion(self, condition: bool, message: str, stacklevel: int = 2) -> None:
        """Assert a condition and log the exception if it fails.

        Usage:
        >>> assert_condition(x == 0, f"expected x to be 0, got {x}")
        >>> [06/03/23 16:31:58] ERROR [rx_connect/wbaml/utils/exception.py] [assert_condition: 27] expected x to be 0,
            got 1

        ╭────────────────────────────────────────────────────── Traceback (most recent call last) ─────────────────────╮
        │ /Users/dvsingxe/ai-lab-RxConnect/rx_connect/pill_validation/test.py:19 in assert_condition                   │
        │                                                                                                              │
        │   16                                           ╭──────────────────────── locals ─────────────────────────╮   │
        │   17 def assert_condition(condition, message): │ condition = False                                       │   │
        │   18 │   try:                                  │         e = AssertionError('expected x to be 0, got 1') │   │
        │ ❱ 19 │   │   assert condition, message         │   message = 'expected x to be 0, got 1'                 │   │
        │   20 │   except AssertionError as e:           ╰─────────────────────────────────────────────────────────╯   │
        │   21 │   │   logger.exception("oopsie")                                                                      │
        │   22 │   │   raise                                                                                           │
        ╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

        Args:
            condition (bool): The condition to assert.
            message (str): The message to log if the condition fails.
            stacklevel (int): The stacklevel to log the exception at.

        Raises:
            AssertionError: If the condition fails.
        """
        try:
            assert condition, message
        except AssertionError as e:
            self.exception(message, stack_info=True, exc_info=True, stacklevel=stacklevel)
            raise e


def setup_logger(log_level: int = logging.INFO) -> CustomLogger:
    """Setup a colored logger with custom formatting."""
    # Create a logger
    logger = CustomLogger(log_level)

    if any(isinstance(handler, (logging.StreamHandler, RichHandler)) for handler in logger.handlers):
        # logger already initialized
        return logger

    # Create a console handler
    handler = CustomRichHandler(
        rich_tracebacks=True,
        show_path=False,
        markup=True,
        omit_repeated_times=False,
        tracebacks_show_locals=True,
    )

    # Add the console handler to the logger
    logger.addHandler(handler)

    return logger
