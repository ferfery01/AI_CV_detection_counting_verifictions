from typing import List

import click
import torch
from super_gradients.common.data_types.enum import MultiGPUMode
from super_gradients.training import Trainer, models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback,
)
from super_gradients.training.utils.distributed_training_utils import setup_device

from rx_connect import CKPT_DIR
from rx_connect.detection.dataset import create_dataloaders

"""Script to train a YOLO_NAS model."""


@click.command()
@click.option(
    "-m",
    "--yolo-model",
    default="yolo_nas_l",
    type=click.Choice(["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"]),
    show_default=True,
    help="Model to train.",
)
@click.option(
    "-d", "--data-dir", default="/media/RxConnectShared/synthetic", required=True, help="Path to the dataset."
)
@click.option(
    "-e",
    "--experiment_name",
    required=True,
    help="""Name of the experiment which is used for logging and loading purposes.
    The model name gets appended to the experiment name.
    """,
)
@click.option(
    "-i",
    "--input-dim",
    nargs=2,
    type=int,
    default=[640, 640],
    help="Input dimension of the model.",
)
@click.option(
    "-c",
    "--classes",
    multiple=True,
    type=str,
    default=["Pill"],
    show_default=True,
    help="Classes to train on.",
)
@click.option(
    "-r",
    "--resume",
    is_flag=True,
    help="""Whether to continue training from ckpt with the same experiment name
    (i.e resume from CKPT_ROOT_DIR/EXPERIMENT_NAME/ckpt_latest.pth)
    """,
)
@click.option("-b", "--batch-size", default=20, show_default=True, help="Batch size.")
@click.option("-nw", "--num-workers", default=8, show_default=True, help="Number of workers.")
@click.option("-ne", "--num-epochs", default=15, show_default=True, help="Number of epochs.")
@click.option(
    "-st",
    "--score-thresh",
    default=0.5,
    show_default=True,
    help="Score threshold for the metrics.",
)
@click.option(
    "-it",
    "--iou-thresh",
    default=0.7,
    show_default=True,
    help="IoU threshold below which all overlapping boxes are discarded during NMS.",
)
@click.option("--lr", default=5e-4, show_default=True, help="Learning rate.")
@click.option(
    "-wlre",
    "--lr-warmup-epochs",
    type=int,
    default=3,
    show_default=True,
    help="Number of epochs to warmup the learning rate.",
)
@click.option(
    "-wlr",
    "--lr-warmup-initial-lr",
    type=float,
    default=1e-6,
    show_default=True,
    help="Initial learning rate for the warmup.",
)
@click.option(
    "-o",
    "--optimizer",
    default="AdamW",
    type=click.Choice(["SGD", "Adam", "AdamW"]),
    help="Optimizer to use.",
)
@click.option("-mp", "--mixed-precision", is_flag=True, help="Use mixed precision training.")
def main(
    yolo_model: str,
    data_dir: str,
    experiment_name: str,
    input_dim: List[int],
    classes: List[str],
    resume: bool,
    batch_size: int,
    num_workers: int,
    num_epochs: int,
    score_thresh: float,
    iou_thresh: float,
    lr: float,
    lr_warmup_epochs: int,
    lr_warmup_initial_lr: float,
    optimizer: str,
    mixed_precision: bool,
) -> None:
    # Check if experiment name is valid
    if ":" in experiment_name:
        raise ValueError("Experiment name cannot contain ':'.")

    # Append model name to experiment name
    experiment_name = f"{experiment_name}:{yolo_model}"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    if mixed_precision and device == "cpu":
        raise ValueError("Mixed precision training is not supported on CPU.")

    # Setup device
    setup_device(device=device, multi_gpu=MultiGPUMode.DATA_PARALLEL)

    # Load dataloaders
    train_data, val_data = create_dataloaders(data_dir, classes, input_dim, batch_size, num_workers)

    # Initialize training parameters
    n_classes: int = len(classes)
    train_params = {
        "resume": resume,
        "average_best_models": True,
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": lr_warmup_initial_lr,
        "lr_warmup_epochs": lr_warmup_epochs,
        "initial_lr": lr,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.01,
        "optimizer": optimizer,
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": num_epochs,
        "mixed_precision": mixed_precision,
        "loss": PPYoloELoss(
            num_classes=n_classes,
            use_static_assigner=False,
            reg_max=16,
            classification_loss_weight=0.5,
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                num_cls=n_classes,
                # Non-maximum Suppression (NMS) Module
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=100,
                    nms_threshold=iou_thresh,
                ),
                normalize_targets=True,
                score_thres=score_thresh,
                top_k_predictions=100,
            )
        ],
        "metric_to_watch": "mAP@0.50",
    }

    # Use the trainer to train the model
    trainer = Trainer(experiment_name=experiment_name, ckpt_root_dir=f"{CKPT_DIR}/detection")
    model = models.get(yolo_model, num_classes=n_classes, pretrained_weights="coco")
    trainer.train(model=model, training_params=train_params, train_loader=train_data, valid_loader=val_data)


if __name__ == "__main__":
    main()
