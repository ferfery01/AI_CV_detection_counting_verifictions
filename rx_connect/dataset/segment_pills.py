from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Union

import click
import torch
import torchvision.transforms.functional as TF
from joblib import Parallel, delayed
from torchvision.ops import masks_to_boxes
from torchvision.utils import save_image
from tqdm import tqdm

from rx_connect.core.images.geometry import expand_bounding_box, extract_ROI
from rx_connect.core.images.io import load_image
from rx_connect.core.images.masks import fill_largest_contour, generate_grayscale_mask
from rx_connect.core.images.transforms import resize_and_center
from rx_connect.core.utils.str_utils import str_to_hash
from rx_connect.dataset.utils import load_consumer_image_df_by_layout
from rx_connect.tools.logging import setup_logger

logger = setup_logger()

"""This script is used to segment all the pill belonging to the `MC_C3PI_REFERENCE_SEG_V1.6` layout.
The script will crop the metadata from the bottom of the image, fill the largest contour in the mask,
expand the bounding box by a specified number of pixels, and resize the image and mask to a specific
size. The cropped image and mask will be saved to the output directory.
"""


def segment_pills(image_path, image_dir, mask_dir, height, width, expand_pixels, bottom_crop_pixels) -> None:
    """Segments the pills in the image and saves the cropped image and mask to the output directory."""
    image = load_image(image_path)
    height, width = image.shape[:2]

    # Generate a grayscale mask for the image
    mask = generate_grayscale_mask(image)

    # Crop the metadata from the bottom of the image
    image, mask = image[:-bottom_crop_pixels, :], mask[:-bottom_crop_pixels, :]

    # Fill the largest contour in the mask
    mask = fill_largest_contour(mask, fill_value=1)

    # Convert images and masks to tensors
    image_t = torch.tensor(image).permute(2, 0, 1)
    mask_t = torch.tensor(mask).unsqueeze(0)

    # Get the bounding box for the mask
    bbox = masks_to_boxes(mask_t).squeeze(0)

    # Expand the bounding box by the specified number of pixels
    bbox = expand_bounding_box(bbox, (height, width), expand_pixels)

    # Extract the patch from the image and mask
    image_patch = extract_ROI(image_t, bbox)
    mask_patch = extract_ROI(mask_t, bbox)

    # Resize the image and mask to a specific size
    image_patch = resize_and_center(image_patch, height, width)
    mask_patch = resize_and_center(mask_patch, height, width, interpolation=TF.InterpolationMode.NEAREST)

    # Save the image and mask to the output directory
    hash_str = str_to_hash(image_path.name)
    save_image(image_patch.float() / 255, image_dir / f"{hash_str}.jpg")
    save_image(mask_patch.float(), mask_dir / f"{hash_str}.png")
    """Masks are saved as PNG format as it can handle binary data without loss of information.
    """


@click.command()
@click.option(
    "-d",
    "--data-dir",
    default="./data/Pill_Images",
    type=click.Path(exists=True),
    help="Path to the directory containing the consumer grade images and the associated csv file.",
)
@click.option(
    "-o",
    "--output-dir",
    default="./data/Pill_Images/segment",
    help="Path to the output directory where the new cropped images and masks will be saved.",
)
@click.option(
    "-b",
    "--bottom-crop-pixels",
    default=340,
    show_default=True,
    help="Number of pixels to crop from the bottom of the image to remove the metadata.",
)
@click.option(
    "-e",
    "--expand-pixels",
    default=50,
    show_default=True,
    help="Number of pixels to expand the bounding box.",
)
@click.option("-h", "--height", default=720, show_default=True, help="Pill image height in pixels.")
@click.option("-w", "--width", default=1280, show_default=True, help="Pill image width in pixels.")
@click.option(
    "-c",
    "--num-cpu",
    default=cpu_count() // 2,
    show_default=True,
    help="The number of CPU cores to use. Use 1 for debugging.",
)
def main(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    bottom_crop_pixels: int,
    expand_pixels: int,
    height: int,
    width: int,
    num_cpu: int,
) -> None:
    layout = "MC_C3PI_REFERENCE_SEG_V1.6"
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    original_images_dir = data_dir / "images" / layout

    # Create image and mask directories
    image_dir, mask_dir = output_dir / "images", output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Load all the appropriate Reference Segmentation Layout images
    df = load_consumer_image_df_by_layout(data_dir, layout)

    # Filter out images that have already been segmented or don't exist
    images_to_segment = [
        original_images_dir / image_name
        for image_name in df.FileName.unique()
        if (original_images_dir / image_name).exists()
        and not (mask_dir / f"{str_to_hash(image_name)}.png").exists()
    ]
    logger.info(f"Found {len(images_to_segment)} images to segment.")

    # Generate images and annotations and save them
    kwargs = {
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "height": height,
        "width": width,
        "expand_pixels": expand_pixels,
        "bottom_crop_pixels": bottom_crop_pixels,
    }
    Parallel(n_jobs=num_cpu)(
        delayed(partial(segment_pills, **kwargs))(image_path)
        for image_path in tqdm(images_to_segment, desc="Segmenting pills")
    )


if __name__ == "__main__":
    main()
