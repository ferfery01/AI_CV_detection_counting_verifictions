from pathlib import Path
from typing import List, Union

import click
import cv2
import numpy as np
from joblib import Parallel, delayed
from skimage.io import imread
from tqdm import tqdm

from rx_connect.core.images.grid import create_square_grid, unpack_images_from_grid
from rx_connect.core.utils.func_utils import batch_generator
from rx_connect.generator.grabcut_utils import (
    apply_grabcut,
    create_initial_mask_for_grabcut,
    post_process_mask,
)
from rx_connect.generator.io_utils import get_unmasked_image_paths

_PAD_WIDTH = 15
"""The number of pixels to pad the images on each side when creating a grid of images.
"""


def generate_final_mask(
    image_paths: List[Path],
    output_masks_path: Path,
    use_sam: bool = False,
    grids: int = 3,
    thresh: float = 0.2,
    num_iter: int = 10,
) -> None:
    """Generate the segmentation mask for each image. The mask can be either generated
    using the SAM-HQ model or using grabcut. The SAM model use a grid of images to
    generate the mask. The grabcut method uses only a single image.
    """
    # Load all the images from the image paths
    images: List[np.ndarray] = [imread(image_path) for image_path in image_paths]

    # Generate the masks for each image either using the SAM model or using grabcut
    if use_sam:
        # Import here to avoid loading the model if not needed. Otherwise, it will be loaded
        # even if the user is not using the SAM model.
        from rx_connect.generator.sam_utils import get_mask_from_SAM

        # Put the images in a grid and get the combined mask from the SAM model
        images_in_grid = create_square_grid(images, grids, pad_width=_PAD_WIDTH)
        combined_mask = get_mask_from_SAM(images_in_grid)

        # Unpack the images from the grid
        masks_per_image = unpack_images_from_grid(combined_mask, grids, pad_width=_PAD_WIDTH)
    else:
        # Use grabcut and some post-processing to get the final mask
        mask = create_initial_mask_for_grabcut(image_paths[0], threshold=thresh)
        masks_per_image = [apply_grabcut(images[0], mask, num_iter)]

    for idx, image_path in enumerate(image_paths):
        final_mask = post_process_mask(masks_per_image[idx])

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
    help="Use the SAM-HQ model to generate the masks. If not set, grabcut will be used.",
)
@click.option(
    "-g",
    "--grids",
    default=3,
    show_default=True,
    help="""The number of square grids to use when arranging the images for SAM.
    The square grid structure with more than 3 rows and columns will increase the
    speed the mask creation but will decrease the quality of the mask. The default
    value of 3 is a good trade-off between speed and quality. Only used if --use-sam
    is set.""",
)
@click.option(
    "-t",
    "--thresh",
    default=0.2,
    show_default=True,
    help="""Intensity threshold for creating the initial binary mask. Only used if
    --use-sam is not set.""",
)
@click.option(
    "-n",
    "--num-iter",
    default=10,
    show_default=True,
    help="The number of iterations to run grabcut algorithm. Only used if --use-sam is not set.",
)
@click.option(
    "-c",
    "--num-cpu",
    default=6,
    show_default=True,
    help="The number of CPU cores to use. Use 1 for debugging.",
)
def main(
    input_images_path: Union[str, Path],
    output_masks_path: Union[str, Path],
    use_sam: bool,
    grids: int,
    thresh: float,
    num_iter: int,
    num_cpu: int,
) -> None:
    # Create the output folder if it doesn't exist
    input_images_path = Path(input_images_path)
    output_masks_path = Path(output_masks_path)
    output_masks_path.mkdir(parents=True, exist_ok=True)

    # Get the images that need to be masked
    images_path = get_unmasked_image_paths(input_images_path, output_masks_path)

    # Batch the images appropriately depending on whether we are using SAM or not
    batch_size = grids**2 if use_sam else len(images_path)
    batched_images_path = list(batch_generator(images_path, batch_size))

    # Generate the masks depending on whether we are using SAM or not
    Parallel(n_jobs=num_cpu)(
        delayed(generate_final_mask)(image_paths, output_masks_path, use_sam, grids, thresh, num_iter)
        for image_paths in tqdm(batched_images_path, desc="Generating masks")
    )


if __name__ == "__main__":
    main()
