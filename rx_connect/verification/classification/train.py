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

from rx_connect.core.types.verification.model import LossWeights
from rx_connect.tools.device import parse_cuda_for_devices
from rx_connect.tools.env_setup import set_max_open_files_limit
from rx_connect.tools.logging import setup_logger
from rx_connect.verification.classification.lightning_model import LightningModel
from rx_connect.verification.classification.model import EmbeddingModel, MultiheadModel
from rx_connect.verification.dataset import PillIDDataModule, load_label_encoder
from rx_connect.verification.split import default_split

logger = setup_logger()

FOLDS_DIR = "folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/"


@click.command()
@click.option("-d", "--data-dir", default="./ePillID_data", show_default=True, help="Path to data")
@click.option(
    "-ex",
    "--expt-name",
    help="Name of the experiment to log to Wandb. Ignored if debug flag is set.",
)
@click.option(
    "-a",
    "--arch",
    type=click.Choice(["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]),
    help="Backbone model architecture",
)
@click.option(
    "-pt/--no-pt",
    "--pretrained/--no-pretrained",
    default=True,
    show_default=True,
    help="Use pretrained weights",
)
@click.option(
    "-P",
    "--pooling",
    default="GAvP",
    type=click.Choice(["GAvP"]),
    help="Pooling method",
)
@click.option(
    "--head-type",
    default="arcface",
    show_default=True,
    type=click.Choice(["arcface", "cosface", "sphereface"]),
    help="Type of margin-based loss head",
)
@click.option("-s", "--scale", default=64, show_default=True, help="Scale for the margin-based loss")
@click.option("-m", "--margin", default=0.5, show_default=True, help="Margin for the margin-based loss")
@click.option(
    "-dr",
    "--dropout",
    default=0.5,
    show_default=True,
    help="Dropout rate",
)
@click.option(
    "-ed",
    "--emb-dim",
    default=2048,
    show_default=True,
    help="Dimension of the embedding feature vector used for metric learning",
)
@click.option(
    "-ad",
    "--add-perspective",
    is_flag=True,
    help="Add perspective transform",
)
@click.option(
    "-nw",
    "--num-workers",
    default=cpu_count() // 2,
    show_default=True,
    help="Number of workers for data loading",
)
@click.option(
    "-st/-nst",
    "--sep-side-train/--no-sep-side-train",
    default=True,
    show_default=True,
    help="Train with side info, i.e. front and back side of the pills will be treated as different classes",
)
@click.option(
    "-cw",
    "--cls-weight",
    default=1.0,
    help="Weight for the classification loss",
)
@click.option(
    "-aw",
    "--angular-weight",
    default=0.0,
    help="Weight for the angular margin-based loss",
)
@click.option(
    "-e",
    "--epochs",
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
    "--lr-patience",
    default=2,
    show_default=True,
    help="Number of epochs to wait before reducing learning rate",
)
@click.option(
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
    "-mm",
    "--metric-monitor",
    default="total_loss",
    type=click.Choice(["total_loss", "acc1", "acc5"]),
    help="Metric to monitor for early stopping and reducing learning rate.",
)
@click.option(
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
    arch: str,
    pretrained: bool,
    pooling: str,
    head_type: str,
    scale: int,
    margin: float,
    dropout: float,
    emb_dim: int,
    add_perspective: bool,
    num_workers: int,
    sep_side_train: bool,
    cls_weight: float,
    angular_weight: float,
    epochs: int,
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
    logger.assertion(
        debug or expt_name is not None, "Experiment name must be provided if debug flag is not set."
    )
    if mixed_precision:
        logger.assertion(torch.cuda.is_available(), "Mixed precision training is only supported on CUDA.")

    # Set the maximum number of open files allowed by the systems
    set_max_open_files_limit()

    # Init GPU training specific parameters
    device_ids = parse_cuda_for_devices(cuda)
    device_msg = (
        f"Using CUDA device(s): {device_ids}"
        if torch.cuda.is_available()
        else "CUDA is not available. Using CPU."
    )
    logger.info(device_msg)

    # Set the seed for reproducibility
    L.seed_everything(seed=42, workers=True)
    cudnn.deterministic = True

    # Init logger
    wandb_logger: Union[Logger, bool] = True
    if not debug:
        wandb_logger = WandbLogger(name=expt_name, project="Pill Verification")

    # Init data paths
    folds_csv_dir = Path(data_dir) / FOLDS_DIR
    label_encoder_file = folds_csv_dir / "label_encoder.pkl"
    all_imgs_csv = folds_csv_dir / "pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv"
    val_imgs_csv = folds_csv_dir / "pilltypeid_nih_sidelbls0.01_metric_5folds_3.csv"
    test_imgs_csv = folds_csv_dir / "pilltypeid_nih_sidelbls0.01_metric_5folds_4.csv"
    df = default_split(all_imgs_csv, val_imgs_csv, test_imgs_csv)

    # Load label encoder
    label_encoder = load_label_encoder(all_imgs_csv, label_encoder_file)

    # Init data module
    augment_kwargs = {"add_perspective": add_perspective}
    datamodule = PillIDDataModule(
        root=Path(data_dir) / "classification_data",
        df=df,
        label_encoder=label_encoder,
        batch_size=batch_size,
        num_workers=num_workers,
        **augment_kwargs,
    )
    n_classes = len(label_encoder.classes_)

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

    # Init model
    emb_model = EmbeddingModel(
        arch=arch,
        pooling=pooling,
        dropout_rate=dropout,
        emb_size=emb_dim,
        pretrained=pretrained,
    )
    model = MultiheadModel(
        emb_model, n_classes, sep_side_train=sep_side_train, head_type=head_type, scale2=scale, m=margin
    )

    # Init optimizer and lr scheduler params
    optimizer_init = {"lr": initial_lr, "weight_decay": 1e-4}
    lr_scheduler_init = {"mode": mode, "factor": lr_factor, "patience": lr_patience}

    # Init lightning model
    loss_weights = LossWeights(cls=cls_weight, angular=angular_weight)
    lightning_model = LightningModel(
        model, monitor, sep_side_train, loss_weights, batch_size, optimizer, optimizer_init, lr_scheduler_init
    )

    # Init trainer
    trainer = L.Trainer(
        accelerator="gpu" if isinstance(device_ids, list) else "cpu",
        callbacks=trainer_callbacks,
        devices=device_ids,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        logger=wandb_logger,
        max_epochs=epochs,
        precision=16 if mixed_precision else 32,
        profiler="simple",
    )

    # Train the model ⚡🚅⚡
    trainer.fit(lightning_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
