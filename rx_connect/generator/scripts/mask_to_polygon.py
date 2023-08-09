from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Union

import click
import cv2
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from rx_connect import ROOT_DIR
from rx_connect.core.types.generator import COCO_LABELS
from rx_connect.generator.io_utils import load_comp_mask_paths
from rx_connect.tools.logging import setup_logger

logger = setup_logger()

"""This script is designed to convert binary masks into COCO (Common Objects in Context)
Polygon representation and save the results to a text file. The conversion involves approximating
the contours of the objects in the binary mask, then representing those contours as polygons.

Main Features:
- Convert binary masks to polygons and save to a text file.
- Utilize `cv2.CHAIN_APPROX_SIMPLE` for contour approximation.
- Control over the polygon approximation accuracy through the `--eps` flag.
- Parallel processing to enhance performance.
- Integration of a progress bar for real-time monitoring.

Usage:
- As a standalone script for converting binary masks for objects like pills (circular, capsules,
    oval shapes, etc.) to polygon representation.
- Can be integrated into larger pipelines for image segmentation, object detection, etc.

Example:
    mask_to_polygon --pill-mask-path /path/to/masks --output-folder /path/to/output --eps 0.001

NOTE: Ensure that the input binary masks are properly segmented, as the script is designed to
process well-segmented binary masks.
"""


def mask_to_polygon(mask_path: Path, output_path: Path, eps: float) -> None:
    """Converts a binary mask to COCO Polygon representation and saves it to a txt file.

    Args:
        mask_path (Path): the path to the mask file
        output_path (Path): the path to the output folder where the polygon txt file will be saved
        eps (float): the approximation accuracy of the polygon
    """
    # Load the mask data and get the number of pills
    mask = np.load(mask_path)
    n_pills = np.max(mask)
    heigth, width = mask.shape

    # Loop through each pill in the mask and convert it to a polygon annotation
    annotations: List[str] = []
    for idx in range(1, n_pills + 1):
        # Get the mask of pill at index `idx`
        mask_per_pill = (mask == idx).astype(np.uint8)

        # find the contours of the mask
        contours, _ = cv2.findContours(mask_per_pill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Corner Case: OpenCV detects Contour and Noise.
        # Take the contour with the max number of points, bc noise have less points than contours
        contours_best = max(contours, key=len) if len(contours) > 1 else contours[0]

        # Approximate the contour with a polygon
        poly_approx = approximate_contours(contours_best, eps)

        # Convert the polygon to a string by normalizing the x y coordinates
        string_contrainer = " ".join(f"{point[0]/width} {point[1]/heigth}" for point in poly_approx)
        annotations.append(f"0 {string_contrainer}")

    # Save the polygons to a txt file
    with (output_path / COCO_LABELS / f"{mask_path.stem}.txt").open("w") as f:
        f.write("\n".join(annotations))


def approximate_contours(contour: np.ndarray, eps: float = 0.0015) -> np.ndarray:
    """Approximate the contour with a polygon

    Args:
        contour (np.ndarray): the contour to approximate, (x, y coordinates)
        eps (float, optional): the approximation accuracy of the polygon. Higher values
            will give a more approximate, simpler polygon, while lower values will give
            a polygon closer to the original contour. Defaults to 0.0015.

    Returns:
        np.ndarray: the approximated contour, (x, y coordinates)
    """
    # Approximate the contour with a polygon
    epsilon = eps * cv2.arcLength(contour, True)
    poly_approx = cv2.approxPolyDP(contour, epsilon, True)

    return poly_approx.squeeze()


@click.command()
@click.option(
    "-d",
    "--data-dir",
    default=ROOT_DIR / "data" / "synthetic" / "segmentation",
    type=click.Path(exists=True),
    show_default=True,
    help="Path to the folder containing images along with the composite masks",
)
@click.option(
    "-o",
    "--output-folder",
    default=ROOT_DIR / "data" / "synthetic" / "segmentation",
    type=click.Path(),
    show_default=True,
    help="Path to the output folder where the polygon txt file will be saved",
)
@click.option(
    "-e",
    "--eps",
    default=0.001,
    show_default=True,
    help="""The approximation accuracy of the polygon. Higher values will give a more approximate,
    simpler polygon, while lower values will give a polygon closer to the original contour.""",
)
@click.option(
    "-c",
    "--num-cpu",
    default=cpu_count() // 2,
    show_default=True,
    help="The number of CPU cores to use",
)
def main(data_dir: Union[str, Path], output_folder: Union[str, Path], eps: float, num_cpu: int) -> None:
    # Load all the mask paths
    mask_paths = load_comp_mask_paths(data_dir)

    # Create output folder
    output_path = Path(output_folder)
    (output_path / COCO_LABELS).mkdir(parents=True, exist_ok=True)

    # Generate COCO format labels and save them
    Parallel(n_jobs=num_cpu)(
        delayed(mask_to_polygon)(mask_path, output_path, eps)
        for mask_path in tqdm(mask_paths, desc="Generating COCO labels")
    )
    logger.info(f"Polygon labels (txt) are saved to the folder: {output_path / COCO_LABELS}")


if __name__ == "__main__":
    main()
