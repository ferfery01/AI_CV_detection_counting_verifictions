from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Union

import click
import cv2
import numpy as np
from joblib import Parallel, delayed

from rx_connect import ROOT_DIR
from rx_connect.generator.io_utils import COCO_LABELS, load_comp_mask_paths
from rx_connect.tools.logging import setup_logger

logger = setup_logger()


def mask2polygon(
    pill_mask_paths: Path,
    output_path: Path,
) -> None:
    """
    This function converts mask to polygon, which fits the COCO-seg128's label, in a txt format

    Args:
        mask (hxwx3):  mask with each pill label index as (1,2,...,idx)
    Returns:
        polygon ([xy, xy, ...]):
        a series of normalized x y coordinates representing the polygon of the mask
    """

    mask_output_file_name = pill_mask_paths.stem
    mask_data = np.load(pill_mask_paths)
    mask_pill_num = np.max(mask_data)

    string_contrainer = ""
    for i in range(1, mask_pill_num + 1):  # the mask idx start from 1 to n_pills
        mask_one_pill_position = (mask_data == i).astype(np.uint8)
        # find the contours of the mask
        contours_tmp, _ = cv2.findContours(mask_one_pill_position, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # corner case, opencv detects contour and noise.
        # take the contour with the max number of points, bc noise have less points than contours
        if len(contours_tmp) > 1:
            contours_tmp_best = max(contours_tmp, key=len)
            approx_contours = approximate_contours(contours_tmp_best)

        # if no noise, just take the first element, which is the contour
        else:
            approx_contours = approximate_contours(contours_tmp[0])

        string_contrainer += "0 "  # For now, each contour all have label index as 0
        for item in approx_contours:
            string_contrainer += f"{item[0]} {item[1]} "
        string_contrainer += "\n"

    # save the polygon xy coordinates to the txt file
    with (output_path / COCO_LABELS / f"{mask_output_file_name}.txt").open("w") as f:
        f.write(string_contrainer)


def approximate_contours(contours: List[List[int]]) -> np.ndarray:
    """Approximate the contour with a polygon

    Args:
        List[List[int]: the original contours from the pill mask

    Returns:
        np.ndarray: the approximated contours, (x, y coordinates)
    """

    # Approximate the contour with a polygon
    epsilon = 0.002 * cv2.arcLength(contours, True)  # Adjust the epsilon value as needed
    approx = cv2.approxPolyDP(contours, epsilon, True)

    return approx.squeeze()


@click.command()
@click.option(
    "-p",
    "--pill-mask-path",
    default=ROOT_DIR / "data" / "synthetic" / "segmentation",
    type=click.Path(exists=True),
    show_default=True,
    help="Path to the folder containing composed image masks.",
)
@click.option(
    "-o",
    "--output-folder",
    default=ROOT_DIR / "data" / "synthetic" / "segmentation",
    type=click.Path(),
    show_default=True,
    help="Path to the output folder",
)
@click.option(
    "-c",
    "--num-cpu",
    default=cpu_count() // 2,
    show_default=True,
    help="The number of CPU cores to use",
)
def main(
    pill_mask_path: Union[str, Path],
    output_folder: Union[str, Path],
    num_cpu: int,
):
    # Load composition pill mask paths
    pill_mask_paths = load_comp_mask_paths(pill_mask_path)

    # Create output folder
    output_path = Path(output_folder)
    (output_path / COCO_LABELS).mkdir(parents=True, exist_ok=True)

    # Generate COCO format labels and save them
    Parallel(n_jobs=num_cpu)(
        delayed(mask2polygon)(
            mask_path,
            output_path,
        )
        for mask_path in pill_mask_paths
    )

    logger.info(f"Polygon labels (txt) are saved to the folder:  {output_path} / {COCO_LABELS}")


if __name__ == "__main__":
    main()
