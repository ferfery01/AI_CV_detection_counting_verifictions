from typing import Any, Dict, List, Tuple

import lightning as L
import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy

from rx_connect.core.types.verification.dataset import ePillIDDatasetBatch
from rx_connect.core.types.verification.model import (
    LossWeights,
    MultiheadLossType,
    MultiHeadModelOutput,
)
from rx_connect.tools.logging import setup_logger
from rx_connect.verification.classification.losses import MultiHeadLoss
from rx_connect.verification.classification.model import MultiheadModel

logger = setup_logger()


class LightningModel(L.LightningModule):
    def __init__(
        self,
        model: MultiheadModel,
        monitor: str,
        sep_side_train: bool,
        loss_weights: LossWeights,
        batch_size: int,
        optimizer: str,
        optimizer_init: Dict[str, Any],
        lr_scheduler_init: Dict[str, Any],
    ):
        super().__init__()
        self.n_classes = model.get_original_n_classes()
        self.monitor = monitor
        self.sep_side_train = sep_side_train
        self.loss_weights = loss_weights
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init

        # Save hyperparameters to self.hparams (auto-logged by WandbLogger)
        self.save_hyperparameters(
            "sep_side_train", "loss_weights", "batch_size", "optimizer", "optimizer_init", "lr_scheduler_init"
        )

        # Define model
        self.model = model

        # Metrics
        self.acc1 = Accuracy(top_k=1, num_classes=self.n_classes)
        self.acc5 = Accuracy(top_k=5, num_classes=self.n_classes)

        # Loss function
        self.criterion = MultiHeadLoss(self.n_classes, self.sep_side_train, self.loss_weights)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> MultiHeadModelOutput:
        if self.loss_weights["angular"] > 0:
            return self.model(x, target=labels)
        else:
            return self.model(x)

    def _compute_logits(self, outputs: MultiHeadModelOutput) -> Dict[str, torch.Tensor]:
        """Compute different logits from the model outputs."""
        logits = {}
        if outputs["cls_logits"] is not None:
            logits["cls"] = outputs["cls_logits"]
        if outputs["angular_logits"] is not None:
            logits["angular"] = outputs["angular_logits"]
        if outputs["cls_logits"] is not None and outputs["angular_logits"] is not None:
            logits["total"] = torch.mean(
                torch.stack([outputs["cls_logits"], outputs["angular_logits"]]), dim=0
            )
        else:
            logits["total"] = (
                outputs["cls_logits"] if outputs["cls_logits"] is not None else outputs["angular_logits"]
            )

        return logits

    def _log_metrics(
        self,
        labels: torch.Tensor,
        losses: MultiheadLossType,
        logits: Dict[str, torch.Tensor],
        prefix: str,
        **kwargs,
    ) -> None:
        """Log losses and metrics to the logger."""
        for key, value in losses.items():
            if value is not None:
                self.log(f"{prefix}_{key}_loss", value, **kwargs)  # type: ignore

        for key, value in logits.items():
            self.log(f"{prefix}_{key}_acc1", self.acc1(value, labels), **kwargs)  # type: ignore
            self.log(f"{prefix}_{key}_acc5", self.acc5(value, labels), **kwargs)  # type: ignore

    def _compute_step(self, batch: ePillIDDatasetBatch, prefix: str) -> torch.Tensor:
        inputs, labels, is_front, is_ref = batch["image"], batch["label"], batch["is_front"], batch["is_ref"]
        outputs = self(inputs, labels)

        # Compute loss
        losses = self.criterion(outputs, labels, is_front, is_ref)

        # Compute logits
        logits = self._compute_logits(outputs)

        # Shift back the logits to the original classes, if needed
        if self.sep_side_train:
            for key, value in logits.items():
                logits[key] = self.model.shift_label_indexes(value)

        # Log loss and metric
        log_args = {
            "prog_bar": True,
            "logger": True,
            "on_step": True,
            "on_epoch": True,
            "batch_size": self.batch_size,
        }
        self._log_metrics(labels, losses, logits, prefix, **log_args)

        return losses["total"]

    def training_step(self, batch: ePillIDDatasetBatch, batch_idx: int) -> torch.Tensor:
        return self._compute_step(batch, "train")

    def validation_step(self, batch: ePillIDDatasetBatch, batch_idx: int) -> torch.Tensor:
        return self._compute_step(batch, "val")

    def test_step(self, batch: ePillIDDatasetBatch, batch_idx: int) -> torch.Tensor:
        return self._compute_step(batch, "test")

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Any]]]:
        """Configure optimizer and learning rate scheduler."""
        if self.optimizer == "AdamW":
            optimizer: torch.optim.Optimizer = AdamW(self.parameters(), **self.optimizer_init)
        elif self.optimizer == "Adam":
            optimizer = Adam(self.parameters(), **self.optimizer_init)
        elif self.optimizer == "sgd":
            optimizer = SGD(self.parameters(), **self.optimizer_init)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}. Please choose from Adam, AdamW, sgd")

        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(optimizer, **self.lr_scheduler_init),
            "interval": "epoch",
            "frequency": 1,
            "monitor": self.monitor,
            "strict": True,
        }

        return [optimizer], [lr_scheduler_config]
