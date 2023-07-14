from typing import Any, Dict, List, Tuple

import lightning as L
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy

from rx_connect.core.types.verification.dataset import ePillIDDatasetBatch
from rx_connect.tools.logging import setup_logger
from rx_connect.verification.classification.model import MultiheadModel

logger = setup_logger()


class LightningModel(L.LightningModule):
    def __init__(
        self,
        model: MultiheadModel,
        n_classes: int,
        monitor: str,
        sep_side_train: bool,
        batch_size: int,
        optimizer: str,
        optimizer_init: Dict[str, Any],
        lr_scheduler_init: Dict[str, Any],
        lr: float = 5e-5,
    ):
        super().__init__()
        logger.assertion(lr > 0.0, "Learning rate should be greater than 0.0")

        self.n_classes = n_classes
        self.monitor = monitor
        self.sep_side_train = sep_side_train
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init

        # Save hyperparameters to self.hparams (auto-logged by WandbLogger)
        self.save_hyperparameters("lr", "batch_size", "optimizer", "optimizer_init", "lr_scheduler_init")

        # Define model
        self.model = model

        # Metrics
        self.acc1 = Accuracy(top_k=1, num_classes=self.n_classes)
        self.acc5 = Accuracy(top_k=5, num_classes=self.n_classes)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)

    def _compute_step(self, batch: ePillIDDatasetBatch, prefix: str) -> torch.Tensor:
        inputs, labels = batch["image"], batch["label"]
        logits = self(inputs)["logits"]

        if self.sep_side_train:
            # front/back is treated as different classes
            logits = self.model.shift_label_indexes(logits)

        acc1 = self.acc1(logits, labels)
        acc5 = self.acc5(logits, labels)

        # Compute loss
        loss = self.criterion(logits, labels)

        # Log loss and metric
        log_args = {
            "prog_bar": True,
            "logger": True,
            "on_step": True,
            "on_epoch": True,
            "batch_size": self.batch_size,
        }
        self.log(f"{prefix}_loss", loss, **log_args)  # type: ignore
        self.log(f"{prefix}_acc1", acc1, **log_args)  # type: ignore
        self.log(f"{prefix}_acc5", acc5, **log_args)  # type: ignore

        return loss

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
