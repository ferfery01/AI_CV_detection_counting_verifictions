from typing import List

import click
import torch
from ultralytics import YOLO

from rx_connect import PROJECT_DIR

"""Script to train a YOLO8v-segmentation model.
If you need more hyperparameters for training, just check the online hyp.scratch.yaml file.
Or, print via: vars(model) to see all the attributes. model here is a YOLO model from ultralytics library"""


@click.command()
@click.option(
    "-m",
    "--yolo-model",
    default="yolov8n-seg.pt",
    type=click.Choice(["yolov8n-seg.pt"]),
    show_default=True,
    help="""Type of pretrained model to load for training from ultralytics library.""",
)
@click.option(
    "-y",
    "--yaml-dir",
    default=f"{PROJECT_DIR}/segmentation/instance/data.yaml",
    required=True,
    help="""Path to the yaml file for training the model.
    It contains dataset path for train/val, optional test, and class labels.""",
)
@click.option(
    "-i",
    "--input-dim",
    nargs=2,
    type=int,
    default=[640, 640],
    help="""Input image resolution.
    If using non squre resolution, make sure you turn on the rect(angle) mode.""",
)
@click.option(
    "-o", "--optimizer", default="AdamW", type=click.Choice(["SGD", "Adam", "AdamW"]), help="Optimizer."
)
@click.option("--lr0", default=1e-2, show_default=True, help="Initial learning rate.")
@click.option("--lrf", default=1e-4, show_default=True, help="Final OneCycleLR learning rate (lr0*lrf).")
@click.option("-b", "--batch-size", default=64, show_default=True, help="Batch size.")
@click.option("-ne", "--num-epochs", default=300, show_default=True, help="Number of epochs.")
@click.option("-s", "--seed", default=0, help="Fix seed for training.")
@click.option(
    "-cf", "--conf", default=1e-3, help="Confidence threshold for each pixel belongs to a specific class."
)
@click.option("-i", "--iou", default=0.7, show_default=True, help="IOU threshold.")
@click.option("-r", "--rect", is_flag=True, help="Allow the image to be rectangle shape.")
@click.option("-p", "--pretrained", is_flag=True, help="Model parameters are pretrained.")
@click.option("-om", "--overlap-mask", is_flag=True, help="Whether allow overlap masks.")
@click.option("-nms", "--nms", is_flag=True, help="Whether to use non maximun supression")
@click.option("-do", "--dropout", default=0.0, help="Dropout mode.")
@click.option("-sc", "--save-crop", is_flag=False, help="Save cropped results.")
@click.option("-d", "--device", show_default=True, help="GPU for training.")
@click.option("-v", "--verbose", is_flag=False, help="Verbose mode.")
@click.option("-re", "--resume", is_flag=True, help="Resume training from last checkpoint.")
@click.option("-sd", "--save-dir", default="./seg_results/", help="Save folder.")
def main(
    yolo_model: str,
    yaml_dir: str,
    input_dim: List[int],
    rect: bool,
    conf: float,
    iou: float,
    num_epochs: int,
    batch_size: int,
    lr0: float,
    lrf: float,
    seed: float,
    optimizer: str,
    pretrained: bool,
    overlap_mask: bool,
    nms: bool,
    dropout: bool,
    save_crop: bool,
    device: str,
    verbose: bool,
    resume: bool,
    save_dir: str,
) -> None:
    # user first specify which cuda; then program check if cuda is available
    device = device if torch.cuda.is_available() else "cpu"

    # Load model
    model = YOLO(yolo_model)

    # Train the model. All the files will be saved automatically.
    _ = model.train(
        data=yaml_dir,
        imgsz=input_dim,
        conf=conf,
        iou=iou,
        rect=rect,
        overlap_mask=overlap_mask,
        nms=nms,
        save_crop=save_crop,
        lr0=lr0,
        lrf=lrf,
        batch=batch_size,
        device=device,
        epochs=num_epochs,
        optimizer=optimizer,
        seed=seed,
        pretrained=pretrained,
        dropout=dropout,
        verbose=verbose,
        resume=resume,
        project=save_dir,
    )


if __name__ == "__main__":
    main()
