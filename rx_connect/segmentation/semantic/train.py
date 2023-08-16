from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Optional, Union

import click
import lightning as L
import torch
import torch.backends.cudnn as cudnn
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import Logger, WandbLogger

from rx_connect.core.utils.func_utils import to_tuple
from rx_connect.segmentation.semantic.datasets import SegDataModule
from rx_connect.segmentation.semantic.model import SegModel
from rx_connect.tools.device import DeviceType, get_device_type, parse_cuda_for_devices
from rx_connect.tools.env_setup import set_max_open_files_limit
from rx_connect.tools.logging import setup_logger

logger = setup_logger()


@click.command()
@click.option("-d", "--data-dir", required=True, help="Directory containing the segmentation dataset.")
@click.option(
    "-ex",
    "--expt-name",
    help="Name of the experiment to log in Wandb. Ignored if debug flag is set.",
)
@click.option(
    "-e",
    "--encoder",
    default="efficientnet-b2",
    type=click.Choice(
        [
            "efficientnet-b0",
            "efficientnet-b1",
            "efficientnet-b2",
            "efficientnet-b3",
            "efficientnet-b4",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "xception",
            "inceptionv4",
            "inceptionresnetv2",
        ]
    ),
    help="Backbone model architecture",
)
@click.option(
    "-i",
    "--image-size",
    nargs=2,
    type=int,
    default=[416, 720],
    show_default=True,
    help="Image size to use for training and validation",
)
@click.option(
    "-nw",
    "--num-workers",
    default=cpu_count() // 2,
    show_default=True,
    help="Number of workers for data loading",
)
@click.option(
    "-ne",
    "--num-epochs",
    default=-1,
    show_default=True,
    help="Number of epochs to train for",
)
@click.option(
    "-b",
    "--batch-size",
    default=256,
    show_default=True,
    help="Batch size",
)
@click.option(
    "-lr",
    "--initial-lr",
    default=1e-4,
    show_default=True,
    help="Initial learning rate",
)
@click.option(
    "-lrp",
    "--lr-patience",
    default=2,
    show_default=True,
    help="Number of epochs to wait before reducing learning rate",
)
@click.option(
    "-lrf",
    "--lr-factor",
    default=0.5,
    show_default=True,
    help="Factor by which to reduce learning rate",
)
@click.option(
    "-o",
    "--optimizer",
    default="AdamW",
    show_default=True,
    type=click.Choice(["Adam", "AdamW", "sgd"]),
    help="Optimizer",
)
@click.option(
    "-p",
    "--patience",
    default=5,
    show_default=True,
    help="Number of epochs to wait before early stopping",
)
@click.option(
    "-m",
    "--metric-monitor",
    default="lovasz_loss",
    type=click.Choice(["lovasz_loss", "dice_loss"]),
    help="Metric to monitor for early stopping and reducing learning rate.",
)
@click.option(
    "-c",
    "--cuda",
    type=str,
    required=True if torch.cuda.is_available() else False,
    help="""
    CUDA device IDs to use for training the model.
    Read the docs for more info:
    https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu_basic.html#choosing-gpu-devices
    """,
)
@click.option(
    "-mp",
    "--mixed-precision",
    is_flag=True,
    help="Use mixed precision training",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Whether to run in debug mode. This will turn off Wandb logging",
)
def main(
    data_dir: Union[str, Path],
    expt_name: str,
    encoder: str,
    image_size: List[int],
    num_workers: int,
    num_epochs: int,
    batch_size: int,
    initial_lr: float,
    lr_patience: int,
    lr_factor: float,
    optimizer: str,
    patience: int,
    metric_monitor: str,
    cuda: Optional[str],
    mixed_precision: bool,
    debug: bool,
) -> None:
    # Validate arguments
    assert debug or expt_name is not None, "Experiment name must be provided if debug flag is not set."
    if mixed_precision and not torch.cuda.is_available():
        logger.warning("Mixed precision training is not available. Ignoring the flag.")
        mixed_precision = False

    # Set the maximum number of open files allowed by the systems
    set_max_open_files_limit()

    # Set the device and device ids
    device = get_device_type()
    device_ids: Union[int, List[int]] = 1
    if device == DeviceType.CUDA:
        device_ids = parse_cuda_for_devices(cuda)

    # Set the seed for reproducibility
    L.seed_everything(seed=42, workers=True)
    cudnn.deterministic = True

    # Init logger
    wandb_logger: Union[Logger, bool] = True
    if not debug:
        wandb_logger = WandbLogger(name=expt_name, project="Pill Verification")

    # Init data module
    datamodule = SegDataModule(
        root_dir=data_dir,
        image_size=to_tuple(image_size),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Init callbacks
    monitor: str = f"val_{metric_monitor}_epoch"
    mode: str = "min" if "loss" in metric_monitor else "max"
    early_stop_callback = EarlyStopping(
        monitor=monitor,
        patience=patience,
        verbose=True,
        mode=mode,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        verbose=True,
        mode=mode,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=True,
    )
    lr_monitor = LearningRateMonitor()
    trainer_callbacks: List[Callback] = [early_stop_callback, checkpoint_callback, lr_monitor]

    # Init optimizer and lr scheduler params
    optimizer_init = {"lr": initial_lr, "weight_decay": 1e-4}
    lr_scheduler_init = {"mode": mode, "factor": lr_factor, "patience": lr_patience}

    # Init lightning model
    lightning_model = SegModel(encoder, monitor, batch_size, optimizer, optimizer_init, lr_scheduler_init)

    # Init trainer
    trainer = L.Trainer(
        accelerator=device.value,
        callbacks=trainer_callbacks,
        devices=device_ids,
        enable_progress_bar=True,
        gradient_clip_val=5.0,
        logger=wandb_logger,
        max_epochs=num_epochs,
        precision=16 if mixed_precision else 32,
        profiler="simple",
    )

    # Train the model ⚡🚅⚡
    trainer.fit(lightning_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
