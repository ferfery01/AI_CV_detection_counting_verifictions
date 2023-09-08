from pathlib import Path
from typing import List, Union

import click
import numpy as np
import torch
from PIL import Image
from super_gradients.training.utils.distributed_training_utils import setup_device
from tqdm import tqdm
from ultralytics import YOLO

from rx_connect import ROOT_DIR, SHARED_REMOTE_DIR
from rx_connect.tools.logging import setup_logger

logger = setup_logger()


def detect_masks(
    img_path: Path,
    pred_dir: Path,
    model: YOLO,
    conf: float,
    show_conf: bool,
    device: str,
) -> None:
    # Load the image
    image = Image.open(img_path)

    # Segment the pills in the image
    pred_results = model(image, conf=conf)

    # Predict the masks and bboxes. Mask is predicted in binary format
    # it uses one image as input, and saves all result in a list, so just take the [0]
    pred_result = pred_results[0]
    pred_mask = pred_result.masks.masks  # N x w x d: (w, d are diff from original size)
    pred_bbox = pred_result.boxes.boxes  # N x 6: x; y; w; d; confidence_score; class probs

    # Convert torch tensor to nparray.
    pred_mask = pred_mask.cpu().numpy()
    pred_bbox = pred_bbox.cpu().numpy()

    # Save masks
    np.save(f"{pred_dir}/mask/{img_path.stem}.npy", pred_mask)

    # Save bboxes
    np.save(f"{pred_dir}/bbox/{img_path.stem}.npy", pred_bbox)


@click.command()
@click.option(
    "-c",
    "--ckpt-path",
    default=f"{SHARED_REMOTE_DIR}/checkpoints/segmentation/best.pt",
    type=str,
    required=True,
    help="Path to the checkpoint.",
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
    help="Path to the test directory (inference data).",
)
@click.option(
    "-p",
    "--pred-dir",
    default=f"{ROOT_DIR}/data/segmentation/preds",
    show_default=True,
    help="Path to the predictions directory (saved results).",
)
@click.option(
    "-sc",
    "--show-conf",
    is_flag=True,
    show_default=True,
    help="Show the confidence of the predictions.",
)
@click.option("-d", "--device", show_default=True, help="GPU for training.")
def main(
    ckpt_path: str,
    conf: float,
    test_dir: str,
    pred_dir: Union[str, Path],
    show_conf: bool,
    device: str,
) -> None:
    # Convert paths to Path objects
    pred_dir = Path(pred_dir)

    # Load the model on the available device
    device = device if torch.cuda.is_available() else "cpu"
    setup_device(device=device)
    model = YOLO(ckpt_path)
    model.val()  # equivalent to eval()

    # Create the directories
    (pred_dir / "mask").mkdir(parents=True, exist_ok=True)
    (pred_dir / "bbox").mkdir(parents=True, exist_ok=True)

    # Iterate over the images in the test directory
    img_paths: List[Path] = list(Path(test_dir).glob("*.png")) + list(Path(test_dir).glob("*.jpg"))
    logger.info(f"Found {len(img_paths)} images in {test_dir}.")

    for img_path in tqdm(img_paths, desc="Segmenting pills"):
        detect_masks(img_path, pred_dir, model, conf, show_conf, device)


if __name__ == "__main__":
    main()
