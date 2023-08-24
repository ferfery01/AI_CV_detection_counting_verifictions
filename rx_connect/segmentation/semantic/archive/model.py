from collections import defaultdict
from typing import Any, Dict, List, Tuple

import lightning as L
import segmentation_models_pytorch as smp
import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SegModel(L.LightningModule):
    def __init__(
        self,
        encoder: str,
        monitor: str,
        batch_size: int,
        optimizer: str,
        optimizer_init: Dict[str, Any],
        lr_scheduler_init: Dict[str, Any],
    ):
        super().__init__()
        self.encoder = encoder
        self.monitor = monitor
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init

        # Save hyperparameters to self.hparams (auto-logged by WandbLogger)
        self.save_hyperparameters()

        # Initialize model
        self.model = smp.DeepLabV3Plus(encoder_name=self.encoder, encoder_weights="imagenet", classes=1)

        # Loss function
        self.loss_function = smp.losses.LovaszLoss(mode="binary", from_logits=True)
        self.dice_soft = smp.losses.DiceLoss(mode="binary", from_logits=True)

        # Variables to store outputs of each step
        self.step_outputs: Dict[str, List[Dict[str, torch.LongTensor]]] = defaultdict(list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        images, masks = batch["image"], batch["mask"]
        images, masks = images.float(), masks.float()
        logits_mask = self(images)

        ## Compute loss
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        lovasz_loss = self.loss_function(logits_mask, masks)
        dice_soft = self.dice_soft(logits_mask, masks)

        ## Compute metrics for some threshold
        # first convert mask values to probabilities, then apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), masks.long(), mode="binary")

        # Log loss and metric
        log_args = {
            "prog_bar": True,
            "logger": True,
            "on_step": True,
            "on_epoch": True,
            "batch_size": self.batch_size,
        }
        self.log(f"{stage}_lovasz_loss", lovasz_loss, **log_args)  # type: ignore
        self.log(f"{stage}_dice_loss", dice_soft, **log_args)  # type: ignore

        # Save step outputs
        stats = {f"{stage}_tp": tp, f"{stage}_fp": fp, f"{stage}_fn": fn, f"{stage}_tn": tn}
        self.step_outputs[f"{stage}"].append(stats)

        return lovasz_loss

    def _shared_epoch_end(self, stage: str) -> None:
        outputs: List[Dict[str, torch.LongTensor]] = self.step_outputs[f"{stage}"]

        # Aggregate step metics
        tp = torch.cat([x[f"{stage}_tp"] for x in outputs])
        fp = torch.cat([x[f"{stage}_fp"] for x in outputs])
        fn = torch.cat([x[f"{stage}_fn"] for x in outputs])
        tn = torch.cat([x[f"{stage}_tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # Dataset IoU means that we aggregate intersection and union over whole dataset and then compute
        # IoU score. The difference between `dataset_iou` and `per_image_iou` scores can be large, if
        # dataset contains images with empty masks.
        # Empty images influence a lot on `per_image_iou` and much less on `dataset_iou`.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        # Compute accuracy
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")

        # Log metrics
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_accuracy": accuracy,
        }
        self.log_dict(metrics, prog_bar=True, on_epoch=True, logger=True)

        # Clear step outputs
        self.step_outputs[stage].clear()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def on_train_epoch_end(self) -> None:
        return self._shared_epoch_end("train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        return self._shared_epoch_end("val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def on_test_epoch_end(self) -> None:
        return self._shared_epoch_end("test")

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
