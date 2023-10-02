from functools import partial
from math import ceil
from pathlib import Path
from typing import Dict, Mapping, NamedTuple, cast

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
from pytorch_msssim import MS_SSIM
from segmentation_models_pytorch.base import SegmentationModel

from rx_connect.core.callbacks import EarlyStopping, TQDMProgressBar
from rx_connect.core.trainer.utils import clip_grads_fn
from rx_connect.core.utils.download_utils import download_and_extract_archive
from rx_connect.core.utils.func_utils import to_tuple
from rx_connect.segmentation.semantic.augments import SegmentTransform
from rx_connect.segmentation.semantic.datasets import DatasetSplit, SegDataset
from rx_connect.segmentation.semantic.losses import FocalTverskyLoss
from rx_connect.segmentation.semantic.metrics import SegmentMetricReducer
from rx_connect.tools.logging import setup_logger

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
"""Suppress all the insecure request warnings from the urllib3 library.
"""

logger = setup_logger()


class SmpConfig(NamedTuple):
    module: SegmentationModel
    pad_divisor: int


SEG_MODELS_CFG: Mapping[str, SmpConfig] = {
    "DeepLabV3": SmpConfig(smp.DeepLabV3, 8),
    "DeepLabV3+": SmpConfig(smp.DeepLabV3Plus, 16),
    "LinkNet": SmpConfig(smp.Linknet, 32),
    "Unet": SmpConfig(smp.Unet, 32),
    "Unet++": SmpConfig(smp.UnetPlusPlus, 32),
}
"""A mapping of segmentation model names to their corresponding SegmentationModel class
and pad_divisor.
"""


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

        # Scale the batch size based on the global batch size
        self.global_batch_size = self.context.get_global_batch_size()
        self.initial_lr = 1e-4 * self.global_batch_size / 64

        # Initialize the model and optimizer
        self.model = self.context.wrap_model(self.configure_model())
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
            ),  # type: ignore
            step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH,
        )

        ## Loss function
        # Focal Tversky loss: Modified Tversky loss that introduces a focusing parameter that places
        # more weight on the false negatives and false positives, helping the model pay more attention
        # to the minority class.
        self.focal_tversky_loss = FocalTverskyLoss(
            alpha=self.loss_cfg.focal_tversky_loss.alpha,
            beta=self.loss_cfg.focal_tversky_loss.beta,
            gamma=self.loss_cfg.focal_tversky_loss.gamma,
            from_logits=False,
        )
        # MS-SSIM loss: Focuses on the pixel level structure information. This helps in achieving a high
        # positive linear correlation between the ground truth and the predicted masks.
        self.ms_ssim_module = MS_SSIM(data_range=1.0, channel=1)

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

    def configure_model(self) -> SegmentationModel:
        """Configure the model and compile it using the `reduce-overhead` mode for faster training."""
        smp_param = self.hparams.smp_config
        try:
            seg_module = SEG_MODELS_CFG[smp_param.model].module
        except KeyError:
            raise KeyError(
                f"Wrong segmentation model name `{smp_param.model}`, supported models: {list(SEG_MODELS_CFG.keys())}"
            )
        model = seg_module(
            encoder_name=smp_param.encoder, encoder_weights=smp_param.encoder_weights, activation="sigmoid"
        )
        opt_model = torch.compile(model)
        return opt_model

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer and learning rate scheduler."""
        optimizer = getattr(torch.optim, self.hparams.optimizer_config.optimizer)

        return optimizer(
            self.model.parameters(),
            lr=self.initial_lr,
            weight_decay=self.hparams.optimizer_config.weight_decay,
        )

    def _calculate_iter_per_epoch(self) -> int:
        """Calculate the number of iterations per epoch."""
        train_data_size = len(self.init_dataset(DatasetSplit.TRAIN))
        return ceil(train_data_size / self.global_batch_size)

    def init_dataset(self, split: DatasetSplit) -> SegDataset:
        """Initialize the dataset for semantic segmentation.

        Args:
            split (DatasetSplit): The type of data split (TRAIN, VAL, TEST).

        Returns:
            SegDataset: The initialized dataset.
        """
        filename = self.data_cfg.filename
        foldername = filename.split(".")[0]
        dataset_dir = f"{self.download_dir}/{foldername}"

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
            pad_divisor=SEG_MODELS_CFG[self.hparams.smp_config.model].pad_divisor,
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
        self, pred_mask: torch.Tensor, target_mask: torch.Tensor, stage: str
    ) -> Dict[str, torch.Tensor]:
        """Compute the loss for the given stage. The loss is a weighted sum of the Focal Tversky loss
        and the MS-SSIM loss.

        Args:
            pred_mask (torch.Tensor): The predicted segmentation mask.
            target_mask (torch.Tensor): The target segmentation mask.
            stage (str): The stage of the training (train or val).

        Returns:
            Dict[str, torch.Tensor]: A dictionary of the computed losses.
        """
        focal_tversky_loss = self.focal_tversky_loss(pred_mask, target_mask)
        ms_ssim = 1 - self.ms_ssim_module(pred_mask, target_mask)
        total_loss = (
            self.loss_cfg.focal_tversky_loss.coef * focal_tversky_loss
            + self.loss_cfg.ms_ssim_loss.coef * ms_ssim
        )
        return {
            f"{stage}_focal_tversky_loss": focal_tversky_loss,
            f"{stage}_ms_ssim_loss": ms_ssim,
            f"{stage}_total_loss": total_loss,
        }

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        batch = cast(Dict[str, torch.Tensor], batch)
        images, target_masks = batch["image"], batch["mask"]
        logit_masks = self.model(images)

        ## Compute loss
        loss_dict = self._compute_loss(logit_masks, target_masks, stage="train")

        # Calculate the gradients with the dice loss
        self.context.backward(loss_dict["train_total_loss"])
        self.context.step_optimizer(self.optimizer, clip_grads=self.clip_grads)  # type: ignore

        # Update the progress bar
        self.progress_bar_callback.train_update(batch_idx)

        return loss_dict

    def evaluate_batch(self, batch: TorchData, batch_idx: int) -> Dict[str, torch.Tensor]:
        batch = cast(Dict[str, torch.Tensor], batch)
        images, target_masks = batch["image"], batch["mask"]
        logit_masks = self.model(images)

        ## Compute loss
        loss_dict = self._compute_loss(logit_masks, target_masks, stage="val")

        # Update the metric reducer and progress bar
        self.reducer.update(logit_masks, target_masks.long())
        self.progress_bar_callback.val_update(batch_idx)

        return loss_dict
