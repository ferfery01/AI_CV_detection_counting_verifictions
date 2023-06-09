from pathlib import Path
from typing import Union

import click
import cv2
import numpy as np
from joblib import Parallel, delayed
from skimage.io import imread
from tqdm import tqdm

from rx_connect.dataset_generator.grabcut_utils import (
    apply_grabcut,
    create_initial_mask_for_grabcut,
    post_process_mask,
)
from rx_connect.dataset_generator.io_utils import get_unmasked_image_paths
from rx_connect.tools.logging import setup_logger

logger = setup_logger()


def generate_final_mask(
    image_path: Path,
    output_masks_path: Path,
    use_sam: bool = False,
    threshold: float = 0.2,
    num_iter: int = 10,
    verbose: bool = False,
) -> None:
    """Generate the final mask for a single image."""
    image: np.ndarray = imread(image_path)

    # If use_sam is True, we use the SAM model to generate the initial mask
    # Otherwise, we use the initialize_mask function
    if use_sam:
        # Import here to avoid loading the model if not needed. Otherwise, it will be loaded
        # even if the user does not want to use it on each cpu core.
        from rx_connect.dataset_generator.sam_utils import (
            get_best_mask_per_images,
            predict_masks,
        )

        masks = predict_masks(image)
        mask = get_best_mask_per_images(masks)[0]
    else:
        mask = create_initial_mask_for_grabcut(image_path, threshold=threshold)

    # Use grabcut and some post-processing to get the final mask
    mask = apply_grabcut(image, mask, num_iter)
    final_mask = post_process_mask(mask)
    if verbose:
        logger.info(f"Saving mask for {image_path.name} to {output_masks_path / image_path.name}")

    # Save the final mask
    cv2.imwrite(str(output_masks_path / image_path.name), final_mask)


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
    type=click.Path(exists=False),
    default=Path("./data/masks/"),
    show_default=True,
    help="The path to the mask folder",
)
@click.option(
    "-s",
    "--use-sam",
    is_flag=True,
    help="Use the SAM model to generate the masks initially",
)
@click.option(
    "-t", "--threshold", default=0.2, help="Intensity threshold for creating the initial binary mask."
)
@click.option(
    "-n",
    "--num-iter",
    default=4,
    show_default=True,
    help="The number of iterations to run grabcut",
)
@click.option(
    "-c",
    "--num-cpu",
    default=6,
    show_default=True,
    help="The number of CPU cores to use. Use 1 for debugging.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enables verbose mode",
)
def main(
    input_images_path: Union[str, Path],
    output_masks_path: Union[str, Path],
    use_sam: bool,
    threshold: float,
    num_iter: int,
    num_cpu: int,
    verbose: bool,
) -> None:
    # Create the output folder if it doesn't exist
    input_images_path = Path(input_images_path)
    output_masks_path = Path(output_masks_path)
    output_masks_path.mkdir(parents=True, exist_ok=True)

    # Get the images that need to be masked
    images_path = get_unmasked_image_paths(input_images_path, output_masks_path)

    # Generate the masks
    Parallel(n_jobs=num_cpu)(
        delayed(generate_final_mask)(image_path, output_masks_path, use_sam, threshold, num_iter, verbose)
        for image_path in tqdm(images_path, desc="Generating masks")
    )


if __name__ == "__main__":
    main()
