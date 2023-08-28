import math
import sys
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
from determined.pytorch import PyTorchCallback, PyTorchTrial, PyTorchTrialContext
from determined.pytorch._reducer import _simple_reduce_metrics as reduce_metrics
from tqdm import tqdm


class TQDMProgressBar(PyTorchCallback):
    """A callback that adds a progress bar to the PyTorchTrial. It prints the progress to stdout using
    the tqdm package and shows up to two progress bars:
    - Train progress: shows the progress of the training. It will pause if validation starts and will resume
        when it ends. It also accounts for multiple validation runs during a single training epoch.
    - Validation progress: only visible during validation; shows the total progress over all
        validation datasets.

    Usage:
        1. Instantiate the TQDMProgressBar callback in the __init__ method of the PyTorchTrial class by providing
        the trial instance and an optional refresh rate:
            self.progress_bar_callback = TQDMProgressBar(trial=trial_instance, refresh_rate=1)

        2. Register the callback within the trial's build_callbacks method:
            def build_callbacks(self) -> Dict[str, PyTorchCallback]:
                return {"progress": self.progress_bar_callback}

        3. Call `train_update(batch_idx)` method in the `train_batch` function to update the training progress:
            self.progress_bar_callback.train_update(batch_idx)

        4. Call `val_update(batch_idx)` method in the `evaluate_batch` function to update the validation progress:
            self.progress_bar_callback.val_update(batch_idx)

    The progress bar will be displayed on the console, providing a clear visual representation of the training
    and validation process, making it easier to track the progress of the model's execution. The progress bar will
    be automatically resumed after a checkpoint is loaded or a trial is resumed, ensuring that the progress bar is
    restored to the state it was in when the checkpoint was saved or the trial was paused.

    Attributes:
        trial: The PyTorchTrial object with which this callback is associated.
        refresh_rate: (Optional) The frequency with which the progress bar is updated.

    Methods:
        train_update(batch_idx: int): Call this method at the end of each training batch to update the progress bar.
        val_update(batch_idx: int): Call this method at the end of each validation batch to update the progress bar.
    """

    def __init__(self, trial: PyTorchTrial, refresh_rate: int = 1) -> None:
        self._trial = trial
        self._context = cast(PyTorchTrialContext, getattr(trial, "context"))
        self._total_train_batches = len(self._trial.build_training_data_loader())
        self._total_val_batches = len(self._trial.build_validation_data_loader())
        self._refresh_rate = refresh_rate

        self._train_progress_bar: Optional[tqdm] = None
        self._val_progress_bar: Optional[tqdm] = None
        self._pbar_metrics: Dict[str, Any] = {}

    @property
    def train_progress_bar(self) -> tqdm:
        if self._train_progress_bar is None:
            raise TypeError(
                f"The `{self.__class__.__name__}._train_progress_bar` reference has not been set yet."
            )
        return self._train_progress_bar

    @train_progress_bar.setter
    def train_progress_bar(self, bar: tqdm) -> None:
        self._train_progress_bar = bar

    @property
    def val_progress_bar(self) -> tqdm:
        if self._val_progress_bar is None:
            raise TypeError(
                f"The `{self.__class__.__name__}._val_progress_bar` reference has not been set yet."
            )
        return self._val_progress_bar

    @val_progress_bar.setter
    def val_progress_bar(self, bar: tqdm) -> None:
        self._val_progress_bar = bar

    @property
    def is_enabled(self) -> bool:
        return self._refresh_rate > 0

    def _init_tqdm(self, desc: str) -> tqdm:
        """Initialize a tqdm progress bar with the given description."""
        return tqdm(desc=desc, disable=not self.is_enabled, leave=True, dynamic_ncols=True, file=sys.stdout)

    def on_training_start(self):
        """Initialize the training progress bar."""
        if self._train_progress_bar is None:
            self.train_progress_bar = self._init_tqdm(desc="Training")

    def on_training_epoch_start(self, epoch_idx: int) -> None:
        """Reset the training progress bar for the new epoch."""
        self.train_progress_bar.reset(self.convert_inf(self._total_train_batches))
        self.train_progress_bar.set_description(f"Epoch {epoch_idx}")

    def on_training_batch_end(self, batch_idx: int) -> None:
        """Update the progress bar with the given batch index."""
        n = batch_idx % self._total_train_batches + 1
        if self._should_update(n, self.train_progress_bar.total):
            self._update_n(self.train_progress_bar, n)

    def train_update(self, batch_idx: int) -> None:
        """Update the progress bar with the given batch index."""
        self.on_training_batch_end(batch_idx)

    def on_training_workload_end(self, avg_metrics: Dict[str, Any], batch_metrics: Dict[str, Any]) -> None:
        """Update the progress bar with the given metrics."""
        if not self.train_progress_bar.disable:
            self._pbar_metrics.update(avg_metrics)
            self.train_progress_bar.set_postfix(self._pbar_metrics)

    def on_validation_start(self) -> None:
        """Initialize the validation progress bar."""
        self.val_progress_bar = self._init_tqdm(desc="Validation")
        self.val_progress_bar.reset(self.convert_inf(self._total_val_batches))

    def on_validation_batch_end(self, batch_idx: int) -> None:
        """Update the progress bar with the given batch index."""
        n = batch_idx + 1
        if self._should_update(n, self.val_progress_bar.total):
            self._update_n(self.val_progress_bar, n)

    def on_validation_end(self, metrics: Dict[str, Any]) -> None:
        """Close the validation progress bar."""
        self.val_progress_bar.close()

    def val_update(self, batch_idx: int) -> None:
        """Update the progress bar with the given batch index."""
        self.on_validation_batch_end(batch_idx)

    def on_validation_epoch_end(self, outputs: List[Any]) -> None:
        """Update the progress bar with the given outputs."""
        if self._train_progress_bar is not None:
            if (metrics := self.get_metrics(outputs)) is not None:
                self._pbar_metrics.update(metrics)
            self.train_progress_bar.set_postfix(self._pbar_metrics)
        self.val_progress_bar.close()

    def _should_update(self, current: int, total: int) -> bool:
        """Determines whether the progress bar should be updated based on the current and total values."""
        return self.is_enabled and (current % self._refresh_rate == 0 or current == total)

    @staticmethod
    def _update_n(pbar: tqdm, value: int) -> None:
        """Update the progress bar with the given value."""
        if not pbar.disable:
            pbar.n = value
            pbar.refresh()

    @staticmethod
    def convert_inf(x: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
        """Converts infinite or NaN values to None for compatibility with tqdm."""
        if x is None or math.isinf(x) or math.isnan(x):
            return None
        return x

    def get_metrics(self, outputs: List[Dict]) -> Optional[Dict[str, Any]]:
        """Reduce the metrics from the outputs of the validation epoch."""
        if len(outputs) == 0:
            return None

        metrics = list(outputs[0].keys())
        eval_reducer = self._trial.evaluation_reducer()
        metric_value: Dict[str, Any] = {}
        for metric in metrics:
            value = np.array([output[metric] for output in outputs])
            reducer = eval_reducer[metric] if isinstance(eval_reducer, dict) else eval_reducer
            metric_value[metric] = reduce_metrics(reducer, metrics=value)

        return metric_value

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the callback as a dictionary. The state is composed of the current epoch and batch
        indices. The state is used to restore the callback after a checkpoint is loaded during training. The
        state is also used to restore the callback after a trial is paused and resumed."""
        return {
            "batch_idx": self._context.current_train_batch(),
            "epoch_idx": self._context.current_train_epoch(),
            "pbar_metrics": self._pbar_metrics,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the state of the callback from the state_dict. After a checkpoint is loaded or a trial is resumed,
        the callback is restored using the state saved in the state_dict. This results in the progress bar being
        restored to the state it was in when the checkpoint was saved or the trial was paused."""
        self._pbar_metrics = state_dict["pbar_metrics"]
        self.on_training_start()
        self.on_training_epoch_start(epoch_idx=state_dict["epoch_idx"])
        self._update_n(self.train_progress_bar, state_dict["batch_idx"] + 1)
        self.train_progress_bar.set_postfix(self._pbar_metrics)
