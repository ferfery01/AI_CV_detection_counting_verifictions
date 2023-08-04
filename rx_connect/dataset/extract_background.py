from pathlib import Path
from typing import List, NamedTuple, Union

import click
import numpy as np
from skimage import io
from tqdm import tqdm

from rx_connect.core.images.io import load_image
from rx_connect.core.utils.str_utils import str_to_hash
from rx_connect.dataset.utils import load_consumer_image_df_by_layout
from rx_connect.pipelines.detection import RxDetection
from rx_connect.pipelines.image import RxVision
from rx_connect.tools.logging import setup_logger

logger = setup_logger()


"""This script is designed to extract a patch of the background from consumer-grade images
from the NIH Pill Image Recognition Challenge dataset. The extracted background patches can
be used to generate synthetic images of pills.

The script uses the YOLO-NAS model to detect the bounding boxes of the pills in the image.
The extreme points of the bounding boxes are used to extract the background patches. The
patches are then filtered based on their size and saved to the output directory.

NOTE: The extracted background patches are not guaranteed to be free of pills. The patches
are dependent on the accuracy of the YOLO-NAS model. Hence, it is recommended to manually
inspect all the patches and remove the ones that contain pills.
"""


class BoundingBoxCoordinates(NamedTuple):
    x_min: int
    x_max: int
    y_min: int
    y_max: int


def get_extreme_points(bounding_boxes: List[List[int]]) -> BoundingBoxCoordinates:
    """Returns the extreme points of all the bounding boxes."""
    return BoundingBoxCoordinates(
        x_min=min([bbox[0] for bbox in bounding_boxes]),
        x_max=max([bbox[2] for bbox in bounding_boxes]),
        y_min=min([bbox[1] for bbox in bounding_boxes]),
        y_max=max([bbox[3] for bbox in bounding_boxes]),
    )


def get_cropped_bg_patch(image: np.ndarray, bbox: BoundingBoxCoordinates) -> List[np.ndarray]:
    """Returns a list of images cropped from the background of the input image."""
    return [image[: bbox.y_min, :], image[bbox.y_max :, :], image[:, : bbox.x_min], image[:, bbox.x_max :]]


def fix_orientation(regions: List[np.ndarray]) -> List[np.ndarray]:
    """Rotates the regions to be horizontally oriented."""
    for idx, region in enumerate(regions):
        h, w = region.shape[:2]
        if h > w:
            regions[idx] = np.transpose(region, (1, 0, 2))

    return regions


@click.command()
@click.option(
    "-d",
    "--data-dir",
    default="./data/Pill_Images",
    type=click.Path(exists=True),
    help="Path to the directory containing the consumer grade images and the associated csv file.",
)
@click.option(
    "-o", "--output-dir", help="Path to the output directory where the background images will be saved."
)
@click.option("-mh", "--min-bg-height", default=720, show_default=True, help="Minimum background height.")
@click.option("-mw", "--min-bg-width", default=1280, show_default=True, help="Minimum background width.")
def main(
    data_dir: Union[str, Path], output_dir: Union[str, Path], min_bg_height: int, min_bg_width: int
) -> None:
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    min_area = min_bg_height * min_bg_width

    # Load all the appropriate consumer grade images
    df = load_consumer_image_df_by_layout(data_dir, "C3PI_Test")

    # Initialize the Detection and Vision objects
    detection_obj = RxDetection()
    vision_obj = RxVision()
    vision_obj.set_counter(detection_obj)

    # Ignore the images that have already been processed or do not exist
    img_paths_to_process = [
        image_path
        for image_path in df.Image.unique()
        if (data_dir / image_path).exists() and not (output_dir / f"{str_to_hash(image_path)}.jpg").exists()
    ]
    logger.info(f"Extracting background from {len(img_paths_to_process)} images.")

    for image_path in tqdm(img_paths_to_process, desc="Extracting background"):
        output_file = output_dir / f"{str_to_hash(image_path)}.jpg"
        try:
            vision_obj.load_image(load_image(data_dir / image_path))
        except Exception:
            logger.error(f"[SKIP] Failed to load image {image_path}.")
            continue

        # Get all the bounding boxes
        bounding_boxes = [bbox.bbox for bbox in vision_obj.bounding_boxes]
        if len(bounding_boxes) == 0:
            logger.error(f"[SKIP] No bounding boxes detected for image {image_path}.")
            continue

        # Extract all the different background regions
        bbox = get_extreme_points(bounding_boxes)
        cropped_regions = get_cropped_bg_patch(vision_obj.image, bbox)

        # Rotate the cropped regions horizontally
        cropped_regions = fix_orientation(cropped_regions)

        # Filter out the cropped regions that are too small
        cropped_regions = [
            region
            for region in cropped_regions
            if region.size > min_area and region.shape[0] < min_bg_height and region.shape[1] < min_bg_width
        ]

        # Save the region with the largest area
        if len(cropped_regions) > 0:
            max_area_idx = np.argmax([region.size for region in cropped_regions])
            logger.info(f"Saving background image for {image_path} to {output_file}.")
            io.imsave(output_file, cropped_regions[max_area_idx])


if __name__ == "__main__":
    main()
