from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Union

import click
import numpy as np
import torch
from joblib import Parallel, delayed
from PIL import Image
from super_gradients.common.data_types.enum import MultiGPUMode
from super_gradients.training import models
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.training.utils.predict import (
    ImageDetectionPrediction,
    ImagesDetectionPrediction,
)
from tqdm import tqdm

from rx_connect import ROOT_DIR
from rx_connect.tools.logging import setup_logger

logger = setup_logger()

"""Script to run inference on a directory of images. The script will save the predictions and the cropped images.

Example:
    $ python inference.py -m yolo_nas_l -ckpt /path/to/checkpoint.pth -c Pill \
        -t /path/to/test/dir -p /path/to/pred/dir -c /path/to/crop/dir -sc
"""


YOLO_NAS_MODELS = Union[models.YoloNAS_S, models.YoloNAS_M, models.YoloNAS_L]


def crop_bounding_boxes(model_pred: ImageDetectionPrediction) -> List[np.ndarray]:
    """Crops all the bounding boxes stored in bboxes_xyxy from an input image array.

    Args:
        model_pred: The YOLO NAS model prediction.

    Returns:
        cropped_images (list of numpy.ndarray): An array of cropped images, each corresponding to a bounding box.
    """
    image = model_pred.image
    bboxes_xyxy = model_pred.prediction.bboxes_xyxy

    cropped_images: List[np.ndarray] = []
    for bbox in bboxes_xyxy:
        # Convert the bounding box coordinates to integers
        xmin, ymin, xmax, ymax = map(int, bbox)

        # Crop the image using the bounding box coordinates
        cropped_image = image[ymin:ymax, xmin:xmax]

        # Add the cropped image to the list of cropped images
        cropped_images.append(cropped_image)

    return cropped_images


def detect_pills(
    img_path: Path,
    pred_dir: Path,
    crop_dir: Path,
    best_model: YOLO_NAS_MODELS,
    conf: float,
    show_conf: bool = False,
) -> None:
    # Load the image
    image = Image.open(img_path)

    # Detect the pills in the image
    model_pred: ImagesDetectionPrediction = best_model.predict(image, conf)
    pred: ImageDetectionPrediction = list(model_pred._images_prediction_lst)[0]

    # Save the predictions
    n_pill = len(pred.prediction.bboxes_xyxy)
    Image.fromarray(pred.draw(show_confidence=show_conf)).save(pred_dir / f"{img_path.stem}_{n_pill}.jpg")

    # Crop the bounding boxes
    img_cropped = crop_bounding_boxes(pred)

    # Save the cropped images
    (crop_dir / img_path.stem).mkdir(parents=True, exist_ok=True)
    for idx, img_crop in enumerate(img_cropped):
        Image.fromarray(img_crop).save(crop_dir / img_path.stem / f"{idx}.jpg")


@click.command()
@click.option(
    "-m",
    "--model",
    default="yolo_nas_l",
    type=click.Choice(["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"]),
    show_default=True,
    help="Type of model to test.",
)
@click.option(
    "-c",
    "--ckpt-path",
    type=str,
    required=True,
    help="Path to the checkpoint.",
)
@click.option(
    "--classes",
    multiple=True,
    type=str,
    default=["Pill"],
    show_default=True,
    help="Classes to train on.",
)
@click.option(
    "--conf",
    default=0.7,
    show_default=True,
    help="Confidence threshold for the predictions.",
)
@click.option(
    "-t",
    "--test-dir",
    default="/media/RxConnectShared/synthetic/test/images",
    help="Path to the test directory.",
)
@click.option(
    "-p",
    "--pred-dir",
    default=f"{ROOT_DIR}/data/detection/preds",
    show_default=True,
    help="Path to the predictions directory.",
)
@click.option(
    "-cd",
    "--crop-dir",
    default=f"{ROOT_DIR}/data/detection/crops",
    show_default=True,
    help="Path to the cropped images directory.",
)
@click.option(
    "-sc",
    "--show-conf",
    is_flag=True,
    show_default=True,
    help="Show the confidence of the predictions.",
)
@click.option(
    "-nc",
    "--num-cpu",
    default=cpu_count() // 2,
    show_default=True,
    help="The number of CPU cores to use. Use 1 for debugging.",
)
def main(
    model: str,
    ckpt_path: str,
    classes: List[str],
    conf: float,
    test_dir: str,
    pred_dir: Union[str, Path],
    crop_dir: Union[str, Path],
    show_conf: bool,
    num_cpu: int,
) -> None:
    # Convert paths to Path objects
    pred_dir, crop_dir = Path(pred_dir), Path(crop_dir)

    # Load the model on the available device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_device(device=device, multi_gpu=MultiGPUMode.DATA_PARALLEL)
    best_model = models.get(model, num_classes=len(classes), checkpoint_path=ckpt_path)
    best_model.eval()

    # Create the directories
    pred_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over the images in the test directory
    img_paths: List[Path] = list(Path(test_dir).glob("*.png")) + list(Path(test_dir).glob("*.jpg"))
    logger.info(f"Found {len(img_paths)} images in {test_dir}.")

    Parallel(n_jobs=num_cpu)(
        delayed(detect_pills)(img_path, pred_dir, crop_dir, best_model, conf, show_conf)
        for img_path in tqdm(img_paths, desc="Detecting pills")
    )


if __name__ == "__main__":
    main()
