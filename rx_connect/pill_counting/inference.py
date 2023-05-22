from pathlib import Path
from typing import List

import click
import numpy as np
import super_gradients
import torch
from PIL import Image
from super_gradients.training import models
from tqdm import tqdm

"""Script to run inference on a directory of images. The script will save the predictions and the cropped images.

Example:
    $ python inference.py -m yolo_nas_l -ckpt /path/to/checkpoint.pth -c Pill \
        -t /path/to/test/dir -p /path/to/pred/dir -c /path/to/crop/dir -sc
"""


def crop_bounding_boxes(image: np.ndarray, bboxes_xyxy: np.ndarray) -> List[np.ndarray]:
    """
    Crops all bounding boxes stored in bboxes_xyxy as np.ndarray with the x and y coordinates of each bounding box
    from an input image array.

    Args:
        image (numpy.ndarray): The input image.
        bboxes_xyxy (numpy.ndarray): An array of bounding box coordinates in the format [x1, y1, x2, y2].

    Returns:
        cropped_images (list of numpy.ndarray): An array of cropped images, each corresponding to a bounding box.
    """
    cropped_images: List[np.ndarray] = []
    for bbox in bboxes_xyxy:
        # Convert the bounding box coordinates to integers
        xmin, ymin, xmax, ymax = map(int, bbox)

        # Crop the image using the bounding box coordinates
        cropped_image = image[ymin:ymax, xmin:xmax]

        # Add the cropped image to the list of cropped images
        cropped_images.append(cropped_image)

    return cropped_images


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
    "-ckpt",
    "--checkpoint-path",
    type=str,
    required=True,
    help="Path to the checkpoint.",
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
    "--conf",
    default=0.7,
    show_default=True,
    help="Confidence threshold for the predictions.",
)
@click.option(
    "-t",
    "--test-dir",
    type=str,
    required=True,
    help="Path to the test directory.",
)
@click.option(
    "-p",
    "--pred-dir",
    type=str,
    default="Preds",
    help="Path to the predictions directory.",
)
@click.option(
    "-c",
    "--crop-dir",
    type=str,
    default="Crops",
    help="Path to the cropped images directory.",
)
@click.option(
    "-sc",
    "--show-conf",
    is_flag=True,
    show_default=True,
    help="Show the confidence of the predictions.",
)
def main(
    model: str,
    checkpoint_path: str,
    classes: List[str],
    conf: float,
    test_dir: str,
    pred_dir: str,
    crop_dir: str,
    show_conf: bool,
) -> None:
    # Load the model on the available device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    super_gradients.setup_device(device=device)
    best_model = models.get(model, num_classes=len(classes), checkpoint_path=checkpoint_path)
    best_model.eval()

    # Create the directories
    Path(pred_dir).mkdir(parents=True, exist_ok=True)
    Path(crop_dir).mkdir(parents=True, exist_ok=True)

    # Iterate over the images in the test directory
    img_paths: List[Path] = list(Path(test_dir).glob("*.png"))
    for img_path in tqdm(img_paths, desc="Detecting pills"):
        # Load the image
        image = Image.open(img_path)

        # Detect the pills in the image
        model_pred = best_model.predict(image, conf)
        pred = list(model_pred._images_prediction_lst)[0]

        # Save the predictions
        n_pill = len(pred.prediction.bboxes_xyxy)
        Image.fromarray(pred.draw(show_confidence=show_conf)).save(
            Path(pred_dir) / f"{img_path.stem}_{n_pill}.png"
        )

        # Crop the bounding boxes
        image_array = pred.image
        bboxes_xyxy = pred.prediction.bboxes_xyxy
        img_cropped = crop_bounding_boxes(image_array, bboxes_xyxy)

        for idx, img_crop in enumerate(img_cropped):
            Image.fromarray(img_crop).save(Path(crop_dir) / f"{img_path.stem}_{idx}.png")


if __name__ == "__main__":
    main()
