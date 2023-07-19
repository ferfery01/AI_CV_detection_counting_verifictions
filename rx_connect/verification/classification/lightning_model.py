from typing import Any, Dict, List, Tuple

import lightning as L
import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy

from rx_connect.core.types.verification.dataset import ePillIDDatasetBatch
from rx_connect.core.types.verification.model import LossWeights, MultiHeadModelOutput
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
        if self.loss_weights["arcface"] > 0:
            return self.model(x, target=labels)
        else:
            return self.model(x)

    def _compute_step(self, batch: ePillIDDatasetBatch, prefix: str) -> torch.Tensor:
        inputs, labels, is_front, is_ref = batch["image"], batch["label"], batch["is_front"], batch["is_ref"]
        outputs = self(inputs, labels)

        # Compute loss
        losses = self.criterion(outputs, labels, is_front, is_ref)

        # Shift back the logits to the original classes, if needed
        logits = outputs["logits"]
        if self.sep_side_train:
            logits = self.model.shift_label_indexes(logits)

        # Log loss and metric
        log_args = {
            "prog_bar": True,
            "logger": True,
            "on_step": True,
            "on_epoch": True,
            "batch_size": self.batch_size,
        }
        for key, value in losses.items():
            if value is not None:
                self.log(f"{prefix}_{key}_loss", value, **log_args)  # type: ignore

        self.log(f"{prefix}_acc1", self.acc1(logits, labels), **log_args)  # type: ignore
        self.log(f"{prefix}_acc5", self.acc5(logits, labels), **log_args)  # type: ignore

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
