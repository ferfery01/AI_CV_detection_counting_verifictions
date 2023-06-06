from pathlib import Path
from typing import List, Set

import click
import cv2
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from rx_connect.dataset_generator.grabcut_utils import apply_grabcut, post_process_mask
from rx_connect.dataset_generator.sam_utils import (
    get_best_mask_per_images,
    predict_masks,
)
from rx_connect.tools.logging import setup_logger

logger = setup_logger()


def generate_final_mask(
    image_path: Path,
    output_masks_path: Path,
    num_iterations: int = 10,
    verbose: bool = False,
):
    image: np.ndarray = cv2.imread(str(image_path))
    masks = predict_masks(image)
    mask_per_image = get_best_mask_per_images(masks)

    mask = apply_grabcut(image, mask_per_image[0], num_iterations)
    final_mask = post_process_mask(mask)
    if verbose:
        logger.info(f"Saving mask for {image_path.name} to {output_masks_path / image_path.name}")

    cv2.imwrite(str(output_masks_path / image_path.name), final_mask)


def images_to_mask(input_images_path: Path, output_masks_path: Path) -> List[Path]:
    input_images: Set[str] = {img_path.name for img_path in input_images_path.glob("*.jpg")}
    output_masks: Set[str] = {mask_path.name for mask_path in output_masks_path.glob("*.jpg")}

    images_to_mask: Set[str] = input_images.difference(output_masks)
    images_to_mask_path: List[Path] = [input_images_path / img_name for img_name in images_to_mask]
    logger.info(f"Found {len(images_to_mask_path)} images to mask.")

    return images_to_mask_path


@click.command()
@click.option(
    "-i",
    "--input-images-path",
    type=click.Path(exists=True),
    default=Path("./data/images/"),
    show_default=True,
    help="The path to the image folder",
)
@click.option(
    "-o",
    "--output-masks-path",
    type=click.Path(exists=True),
    default=Path("./data/masks/"),
    show_default=True,
    help="The path to the mask folder",
)
@click.option(
    "-n",
    "--num-iterations",
    default=4,
    show_default=True,
    help="The number of iterations to run grabcut",
)
@click.option(
    "-c",
    "--num-cpu",
    default=6,
    show_default=True,
    help="The number of CPU cores to use",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enables verbose mode",
)
def main(
    input_images_path: Path,
    output_masks_path: Path,
    num_iterations: int,
    num_cpu: int,
    verbose: bool,
) -> None:
    # Get the images that need to be masked
    images_path = images_to_mask(input_images_path, output_masks_path)

    # Generate the masks
    Parallel(n_jobs=num_cpu)(
        delayed(generate_final_mask)(image_path, output_masks_path, num_iterations, verbose)
        for image_path in tqdm(images_path, desc="Generating masks")
    )


if __name__ == "__main__":
    main()
