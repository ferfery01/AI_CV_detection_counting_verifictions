from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from determined.pytorch import PyTorchCallback, PyTorchTrialContext

from rx_connect.tools.logging import setup_logger

logger = setup_logger()


class EarlyStopping(PyTorchCallback):
    """Monitor a metric and stop training if it doesn't improve after a given patience.

    Args:
        context (PyTorchTrialContext): The context of the current trial.
        monitor (str): The metric to monitor.
        patience (int): Number of epochs with no improvement after which training will be stopped.
            Under the default configuration, one check happends after every training epoch. It
            must be noted that the patience parameter counts the number of validation checks with
            no improvement, and not the number of training epochs.
        mode (str): One of "min" or "max". In "min" mode, training will stop when the metric monitored
            has stopped decreasing; in "max" mode it will stop when the metric monitored has stopped
            increasing.
        min_delta (float): Minimum change in the monitored metric to qualify as an improvement, i.e.
            an absolute change of less than min_delta, will count as no improvement.
        verbose (bool): If True, prints a message for each validation check that doesn't improve
            the metric.
        strict (bool): Whether to crash the training if the monitored metric is not found in the validation
            metrics.
        check_finite (bool): When set to True, stops training if the monitored metric becomes NaN or Inf.
        stopping_threshold (float): Stop training when the monitored metric reaches this threshold.
        divergence_threshold (float): Stop training when the monitored metric diverges from this threshold.

    Raises:
        RuntimeError: If the metric to monitor is not found in the validation metrics.
        ValueError: If the mode is not one of "min" or "max".

    Usage:
        1. Instantiate the `EarlyStopping` callback in the `__init__` method of the `PyTorchTrial` class by
        providing the context instance and the other appropriate parameters.
            self.early_stopping_callback = EarlyStopping(context=self.context, monitor="val_loss", patience=5)

        2. Register the callback within the trial's `build_callbacks` method:
            def build_callbacks(self) -> Dict[str, PyTorchCallback]:
                return {"early_stopping": self.early_stopping_callback}
    """

    mode_dict = {"min": torch.lt, "max": torch.gt}
    order_dict = {"min": "<", "max": ">"}

    def __init__(
        self,
        context: PyTorchTrialContext,
        monitor: str,
        mode: str = "min",
        patience: int = 5,
        min_delta: float = 0.0,
        verbose: bool = True,
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.context = context
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.strict = strict
        self.check_finite = check_finite
        self.stopping_threshold = stopping_threshold
        self.divergence_threshold = divergence_threshold

        # Initialize the internal variables
        self.wait_count = 0
        self.stopped_epoch = 0
        self.counter = 0

        if self.mode not in self.mode_dict:
            raise ValueError(f"Invalid mode: `{mode}`. Must be {', '.join(self.mode_dict.keys())}.")

        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]

    def _evaluate_stopping_criteria(self, current: torch.Tensor) -> Tuple[bool, Optional[str]]:
        """Evaluates the early stopping criteria and returns a boolean indicating whether the training
        should stop and a message explaining the reason for stopping.
        """
        should_stop: bool = False
        reason: Optional[str] = None

        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                )

        return should_stop, reason

    def _improvement_message(self, current: torch.Tensor) -> str:
        """Formats a log message that informs the user about an improvement in the monitored score."""
        if torch.isfinite(self.best_score):
            msg = (
                f"Metric {self.monitor} improved by {abs(self.best_score - current):.3f} >="
                f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
            )
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current:.3f}"
        return msg

    def _validate_condition_metric(self, metrics: Dict[str, Any]) -> bool:
        """Validates whether the metric to monitor is available in the validation metrics."""
        monitor_val = metrics.get(self.monitor)
        if monitor_val is not None:
            return True

        error_msg = (
            f"Early stopping conditioned on metric `{self.monitor}` which is not available."
            " Pass in or modify your `EarlyStopping` callback to use any of the following:"
            f' `{"`, `".join(list(metrics.keys()))}`'
        )

        if self.strict:
            raise RuntimeError(error_msg)

        if self.verbose > 0:
            logger.warning(error_msg)

        return False

    def _run_early_stopping_check(self, metrics: Dict[str, Any]) -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""

        if not self._validate_condition_metric(metrics):
            return None

        current = torch.tensor(metrics[self.monitor].squeeze())
        should_stop, reason = self._evaluate_stopping_criteria(current)

        # stop every ddp process if any world process decides to stop
        if should_stop:
            self.context.set_stop_requested(stop_requested=True)
            self.stopped_epoch = self.context.current_train_epoch()

        if reason and self.verbose and (self.context.distributed.get_rank() == 0):
            logger.info(reason)

    def on_validation_end(self, metrics: Dict[str, Any]) -> None:
        """Checks whether the early stopping condition is met after every validation epoch."""
        self._run_early_stopping_check(metrics)

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the callback as a dictionary."""
        return {
            "wait_count": self.wait_count,
            "stopped_epoch": self.stopped_epoch,
            "best_score": self.best_score,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the state of the callback from the state_dict."""
        self.wait_count = state_dict["wait_count"]
        self.stopped_epoch = state_dict["stopped_epoch"]
        self.best_score = state_dict["best_score"]
