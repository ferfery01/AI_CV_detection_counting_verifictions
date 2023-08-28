from pathlib import Path
from typing import Any, Dict, cast

import segmentation_models_pytorch as smp
import torch
from determined.pytorch import (
    DataLoader,
    PyTorchCallback,
    PyTorchTrial,
    PyTorchTrialContext,
    TorchData,
)

from rx_connect.core.callbacks.progress import TQDMProgressBar
from rx_connect.core.utils.download_utils import download_and_extract_archive
from rx_connect.core.utils.func_utils import to_tuple
from rx_connect.segmentation.semantic.augments import SegmentTransform
from rx_connect.segmentation.semantic.datasets import DatasetSplit, SegDataset


class SegTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        self.hparams = self.context.get_hparams()
        self.data_config = self.context.get_data_config()

        # Data related params
        self.download_dir = self.data_config["download_dir"]
        self.image_size = to_tuple(self.data_config["image_size"])

        self.model = self.context.wrap_model(
            smp.DeepLabV3Plus(encoder_name=self.hparams["encoder"], encoder_weights="imagenet", classes=1)
        )

        self.optimizer = self.context.wrap_optimizer(self.configure_optimizers())

        self.criterion = smp.losses.DiceLoss(mode="binary", from_logits=True)

        # Instantiate TQDMProgressBar callback
        self.progress_bar_callback = TQDMProgressBar(trial=self, refresh_rate=1)

    def build_callbacks(self) -> Dict[str, PyTorchCallback]:
        return {"progress": self.progress_bar_callback}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer and learning rate scheduler."""
        optimizer = getattr(torch.optim, self.hparams["optimizer"])

        return optimizer(
            self.model.parameters(), lr=self.hparams["initial_lr"], weight_decay=self.hparams["weight_decay"]
        )

    def init_dataset(self, split: DatasetSplit) -> SegDataset:
        """Initialize the dataset for semantic segmentation.

        Args:
            split (DatasetSplit): The type of data split (TRAIN, VAL, TEST).

        Returns:
            SegDataset: The initialized dataset.
        """
        filename = self.data_config["filename"]
        foldername = filename.split(".")[0]
        dataset_dir = f"{self.download_dir}/{foldername}/datasets"

        # Check if the dataset directory exists, or if force_download is enabled
        if self.data_config["force_download"] or not Path(dataset_dir).exists():
            # Download and extract the archive if any of the above conditions are met
            download_and_extract_archive(
                url=self.data_config["url"],
                filename=filename,
                download_root=self.download_dir,
                md5=self.data_config.get("md5"),
            )

        tfms = SegmentTransform(
            train=split == DatasetSplit.TRAIN,
            image_size=self.image_size,
            **self.data_config.get("tfm_kwargs", {}),
        )
        return SegDataset(
            root_dir=dataset_dir,
            data_split=split,
            image_size=self.image_size,
            transforms=tfms,
        )

    def build_training_data_loader(self) -> DataLoader:
        train_dataset = self.init_dataset(DatasetSplit.TRAIN)
        return DataLoader(
            train_dataset,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=self.data_config["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

    def build_validation_data_loader(self) -> DataLoader:
        val_dataset = self.init_dataset(DatasetSplit.VAL)
        return DataLoader(
            val_dataset,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=False,
            num_workers=self.data_config["num_workers"],
            pin_memory=True,
            drop_last=False,
        )

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        batch = cast(Dict[str, torch.Tensor], batch)
        images, masks = batch["image"], batch["mask"]
        images, masks = images.float(), masks.float()
        logits_mask = self.model(images)

        loss = self.criterion(logits_mask, masks)

        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)
        self.progress_bar_callback.train_update(batch_idx)

        return {"loss": loss}

    def evaluate_batch(self, batch: TorchData, batch_idx: int) -> Dict[str, Any]:
        batch = cast(Dict[str, torch.Tensor], batch)
        images, masks = batch["image"], batch["mask"]
        images, masks = images.float(), masks.float()
        logits_mask = self.model(images)

        val_loss = self.criterion(logits_mask, masks)
        self.progress_bar_callback.val_update(batch_idx)

        return {"val_loss": val_loss}
