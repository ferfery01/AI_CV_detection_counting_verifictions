from functools import partial
from math import ceil
from pathlib import Path
from typing import Dict, cast

import segmentation_models_pytorch as smp
import torch
import urllib3
from determined.pytorch import (
    DataLoader,
    LRScheduler,
    PyTorchCallback,
    PyTorchTrial,
    PyTorchTrialContext,
    TorchData,
)
from omegaconf import OmegaConf

from rx_connect.core.callbacks import EarlyStopping, TQDMProgressBar
from rx_connect.core.trainer.utils import clip_grads_fn
from rx_connect.core.utils.download_utils import download_and_extract_archive
from rx_connect.core.utils.func_utils import to_tuple
from rx_connect.segmentation.semantic.augments import SegmentTransform
from rx_connect.segmentation.semantic.datasets import DatasetSplit, SegDataset
from rx_connect.segmentation.semantic.metrics import SegmentMetricReducer
from rx_connect.tools.logging import setup_logger

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
"""Suppress all the insecure request warnings from the urllib3 library.
"""

logger = setup_logger()


class SegTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        self.exp_cfg = OmegaConf.create(self.context.get_experiment_config())
        self.hparams = OmegaConf.create(self.context.get_hparams())
        self.data_cfg = OmegaConf.create(self.context.get_data_config())
        self.loss_cfg = self.hparams.loss

        # Data related params
        self.download_dir = self.data_cfg.download_dir
        self.image_size = to_tuple(self.data_cfg.image_size)

        # Initialize the model and optimizer
        self.model = self.context.wrap_model(
            smp.DeepLabV3Plus(encoder_name=self.hparams.encoder, encoder_weights="imagenet", classes=1)
        )
        self.optimizer = self.context.wrap_optimizer(self.configure_optimizers())
        self.clip_grads = partial(clip_grads_fn, max_norm=self.hparams.optimizer_config.gradient_clip_val)

        # Wrap the LR scheduler
        iters_per_epoch = self._calculate_iter_per_epoch()
        self.lr_scheduler = self.context.wrap_lr_scheduler(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.hparams.lr_config.lr_T_max * iters_per_epoch,
                eta_min=self.hparams.lr_config.lr_eta_min,
                last_epoch=self.hparams.lr_config.lr_last_epoch,
            ),
            step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH,
        )

        ## Loss function
        # Focal Loss will handle the class imbalance, and Lovasz Loss will optimize the Jaccard metric.
        self.focal_loss = smp.losses.FocalLoss(
            mode="binary", alpha=self.loss_cfg.focal_loss.alpha, gamma=self.loss_cfg.focal_loss.gamma
        )
        self.lovasz_loss = smp.losses.LovaszLoss(
            mode="binary", per_image=self.loss_cfg.lovasz_loss.per_image, from_logits=True
        )

        # Instantiate all the callbacks
        self.progress_bar_callback = TQDMProgressBar(trial=self, refresh_rate=2)
        monitor_metric = self.exp_cfg.searcher.metric
        mode = "min" if "loss" in monitor_metric else "max"
        self.early_stopping_callback = EarlyStopping(
            context=self.context, monitor=monitor_metric, mode=mode, patience=self.hparams.patience
        )

        # Instantiate the metric reducer
        self.reducer = self.context.wrap_reducer(
            SegmentMetricReducer(), for_training=False, for_validation=True
        )

    def build_callbacks(self) -> Dict[str, PyTorchCallback]:
        return {"progress": self.progress_bar_callback, "early_stopping": self.early_stopping_callback}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer and learning rate scheduler."""
        optimizer = getattr(torch.optim, self.hparams.optimizer_config.optimizer)

        return optimizer(
            self.model.parameters(),
            lr=self.hparams.initial_lr,
            weight_decay=self.hparams.optimizer_config.weight_decay,
        )

    def _calculate_iter_per_epoch(self) -> int:
        """Calculate the number of iterations per epoch."""
        train_data_size = len(self.init_dataset(DatasetSplit.TRAIN))
        batch_size = self.context.get_global_batch_size()
        return ceil(train_data_size / batch_size)

    def init_dataset(self, split: DatasetSplit) -> SegDataset:
        """Initialize the dataset for semantic segmentation.

        Args:
            split (DatasetSplit): The type of data split (TRAIN, VAL, TEST).

        Returns:
            SegDataset: The initialized dataset.
        """
        filename = self.data_cfg.filename
        foldername = filename.split(".")[0]
        dataset_dir = f"{self.download_dir}/{foldername}/datasets"

        # Check if the dataset directory exists, or if force_download is enabled
        if self.data_cfg.force_download or not Path(dataset_dir).exists():
            # Download and extract the archive if any of the above conditions are met
            download_and_extract_archive(
                url=self.data_cfg.url,
                filename=filename,
                download_root=self.download_dir,
                md5=self.data_cfg.md5,
            )

        tfms = SegmentTransform(
            train=split == DatasetSplit.TRAIN,
            image_size=self.image_size,
            **self.data_cfg.get("tfm_kwargs", {}),
        )
        return SegDataset(
            root_dir=dataset_dir,
            data_split=split,
            image_size=self.image_size,
            transforms=tfms,
        )

    def build_training_data_loader(self) -> DataLoader:
        return DataLoader(
            dataset=self.init_dataset(DatasetSplit.TRAIN),
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def build_validation_data_loader(self) -> DataLoader:
        return DataLoader(
            dataset=self.init_dataset(DatasetSplit.VAL),
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def _compute_loss(
        self, logits_mask: torch.Tensor, masks: torch.Tensor, stage: str
    ) -> Dict[str, torch.Tensor]:
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        focal_loss = self.focal_loss(logits_mask, masks)
        lovasz_loss = self.lovasz_loss(logits_mask, masks)
        total_loss = self.loss_cfg.focal_loss.coef * focal_loss + self.loss_cfg.lovasz_loss.coef * lovasz_loss
        return {
            f"{stage}_focal_loss": focal_loss,
            f"{stage}_lovasz_loss": lovasz_loss,
            f"{stage}_total_loss": total_loss,
        }

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        batch = cast(Dict[str, torch.Tensor], batch)
        images, masks = batch["image"], batch["mask"]
        images, masks = images.float(), masks.float()
        logits_mask = self.model(images)

        ## Compute loss
        loss_dict = self._compute_loss(logits_mask, masks, stage="train")

        # Calculate the gradients with the dice loss
        self.context.backward(loss_dict["train_total_loss"])
        self.context.step_optimizer(self.optimizer, clip_grads=self.clip_grads)  # type: ignore

        # Update the progress bar
        self.progress_bar_callback.train_update(batch_idx)

        return loss_dict

    def evaluate_batch(self, batch: TorchData, batch_idx: int) -> Dict[str, torch.Tensor]:
        batch = cast(Dict[str, torch.Tensor], batch)
        images, masks = batch["image"], batch["mask"]
        images, masks = images.float(), masks.float()
        logits_mask = self.model(images)

        ## Compute loss
        loss_dict = self._compute_loss(logits_mask, masks, stage="val")

        ## Compute metrics for some threshold
        # first convert mask values to probabilities, then apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # Update the metric reducer and progress bar
        self.reducer.update(pred_mask, masks)
        self.progress_bar_callback.val_update(batch_idx)

        return loss_dict
