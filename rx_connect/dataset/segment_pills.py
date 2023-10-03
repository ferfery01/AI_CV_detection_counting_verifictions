import os
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional

import albumentations as A
import click
import torchvision.transforms.functional as TF
from joblib import Parallel, delayed
from skimage.segmentation import clear_border
from torchvision.utils import save_image
from tqdm import tqdm

from rx_connect import ROOT_DIR
from rx_connect.core.images.io import load_image
from rx_connect.core.images.masks import (
    fill_largest_contour,
    generate_grayscale_mask,
    separate_pills_and_masks,
)
from rx_connect.core.images.transforms import fix_image_orientation, resize_and_center
from rx_connect.core.images.types import img_to_tensor
from rx_connect.core.utils.str_utils import str_to_hash
from rx_connect.dataset.utils import Layouts, load_consumer_image_df_by_layout
from rx_connect.pipelines.segment import RxSemanticSegmentation
from rx_connect.tools.logging import setup_logger

logger = setup_logger()

"""This script is used to segment all the pill belonging to the `MC_C3PI_REFERENCE_SEG_V1.6` layout.
The script will crop the metadata from the bottom of the image, fill the largest contour in the mask,
expand the bounding box by a specified number of pixels, and resize the image and mask to a specific
size. The cropped image and mask will be saved to the output directory.
"""


def segment_pills(
    image_path: Path,
    layout: Layouts,
    use_sam: bool,
    segmentation_obj: Optional[RxSemanticSegmentation],
    image_dir: Path,
    mask_dir: Path,
    target_height: int,
    target_width: int,
    expand_pixels: int,
) -> None:
    """Segments the pills in the image and saves the cropped image and mask to the output directory."""
    try:
        image = load_image(image_path)
    except OSError:
        logger.error(
            f"Image {image_path} could not be loaded. It might be corrupted and hence it is being "
            "deleted. Please download it again using the `scrape_nih_images` script."
        )
        os.remove(image_path.resolve())
        return None
    image = fix_image_orientation(image)

    # Pre-process the image based on the layout type
    if layout == Layouts.MC_C3PI_REFERENCE_SEG_V1_6:
        # Crop the metadata from the bottom of the image
        image = image[:-340, :]
    elif layout == Layouts.MC_API_RXNAV_V1_3:
        image = image[:-130, :]
    elif layout in (Layouts.MC_COOKED_CALIBRATED_V1_2, Layouts.C3PI_Reference):
        # Crop the borders from the image
        image = A.CenterCrop(height=2200, width=3100)(image=image)["image"]

    # Since MC_C3PI_REFERENCE_SEG_V1_6 layout is already segmented, we directly generate the mask.
    # For other layouts, we use the semantic segmentation model to generate the mask.
    if layout == Layouts.MC_C3PI_REFERENCE_SEG_V1_6:
        # Generate a grayscale mask for the image
        mask = generate_grayscale_mask(image)
    elif use_sam:
        from rx_connect.generator.sam_utils import get_mask_from_SAM

        # Use the SAM-HQ model to generate the mask
        mask = get_mask_from_SAM(image)
    elif segmentation_obj is not None:
        # Segment the image
        mask = segmentation_obj(image, min_size=10000)
        mask = clear_border(mask)
    else:
        raise ValueError(
            "Either the segmentation object or the SAM model must be provided to generate the mask."
        )

    # Separate each pill and mask
    cropped_pills, cropped_masks = separate_pills_and_masks(
        image, mask, expand_pixels, num_pills=layout.max_pills
    )

    for idx, (cropped_pill, cropped_mask) in enumerate(zip(cropped_pills, cropped_masks)):
        # Fill the largest contour in the mask
        cropped_mask = fill_largest_contour(cropped_mask, fill_value=1)

        # Convert the image and mask to tensors
        image_t, mask_t = img_to_tensor(cropped_pill), img_to_tensor(cropped_mask)

        # Resize the image and mask to a specific size
        image_t = resize_and_center(image_t, target_height, target_width)
        mask_t = resize_and_center(
            mask_t, target_height, target_width, interpolation=TF.InterpolationMode.NEAREST
        )

        # Extract the foreground from the image
        fg_image = image_t * mask_t

        # Save the image and mask to the output directory
        hash_str = str_to_hash(image_path.name)
        save_image(fg_image.float() / 255, image_dir / f"{hash_str}_{idx}.jpg")
        save_image(mask_t.float(), mask_dir / f"{hash_str}_{idx}.png")
        """Masks are saved as PNG format as it can handle binary data without loss of information.
        """


@click.command()
@click.option(
    "-l",
    "--image-layout",
    required=True,
    type=click.Choice(Layouts.members()),
    help="""Select the Image layout to download. More information about the layout can be
    obtained at https://data.lhncbc.nlm.nih.gov/public/Pills/RxImageImageLayouts.docx""",
)
@click.option(
    "-d",
    "--data-dir",
    default=ROOT_DIR / "data" / "Pill_Images",
    type=click.Path(exists=True, path_type=Path),
    help="""Path to the directory containing the pill images and the associated csv file from the
    Computational Photography Project for Pill Identification (C3PI).""",
)
@click.option(
    "-o",
    "--output-dir",
    default=ROOT_DIR / "data" / "Pill_Images" / "RxSegment",
    type=click.Path(exists=False, path_type=Path),
    help="""Path to the directory to save the segmented images and masks. The images and masks
    will be saved in the following structure:
        ├── MC_C3PI_REFERENCE_SEG_V1.6
        │   ├── images
        │   │   ├── 00000001.jpg
        │   │   ├── 00000002.jpg
        │   │   ├── ...
        │   ├── masks
        │   │   ├── 00000001.png
        │   │   ├── 00000002.png
        │   │   ├── ...
        │   ├── ...
    """,
)
@click.option(
    "-e",
    "--expand-pixels",
    default=5,
    show_default=True,
    help="Number of pixels to expand the bounding box.",
)
@click.option("-h", "--height", default=720, show_default=True, help="Pill image height in pixels.")
@click.option("-w", "--width", default=1280, show_default=True, help="Pill image width in pixels.")
@click.option("-s", "--use-sam", is_flag=True, help="Use the SAM-HQ model to generate the masks.")
@click.option(
    "-c",
    "--num-cpu",
    default=cpu_count() // 2,
    show_default=True,
    help="The number of CPU cores to use. Use 1 for debugging.",
)
@click.option(
    "--device",
    default="cpu",
    help="""Device to use for inference. If not specified, defaults to the best available device.
    Not applicable for MC_C3PI_REFERENCE_SEG_V1_6 layout.""",
)
def main(
    image_layout: str,
    data_dir: Path,
    output_dir: Path,
    expand_pixels: int,
    height: int,
    width: int,
    use_sam: bool,
    num_cpu: int,
    device: str,
) -> None:
    layout = Layouts[image_layout]
    original_images_dir = data_dir / "images" / layout.name
    output_dir = output_dir / layout.name

    # Create image and mask directories
    image_dir, mask_dir = output_dir / "images", output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Load all the appropriate Reference Segmentation Layout images
    df = load_consumer_image_df_by_layout(data_dir, layout)
    if len(df) == 0:
        raise ValueError(
            f"No images found for the {layout.name} layout. Did you download the data? "
            f"If not, run `python rx_connect.datasets.scrape_nih_images` to download the data first."
        )

    # Filter out images that have already been segmented or don't exist
    images_to_segment = [
        original_images_dir / image_name
        for image_name in df.FileName.unique()
        if (original_images_dir / image_name).exists()
        and not (mask_dir / f"{str_to_hash(image_name)}_0.png").exists()
    ]
    logger.info(f"Found {len(images_to_segment)} images to segment.")

    # Initialize the segmentation object and transforms
    segmentation_obj: Optional[RxSemanticSegmentation] = None
    if not use_sam or layout != Layouts.MC_C3PI_REFERENCE_SEG_V1_6:
        segmentation_obj = RxSemanticSegmentation(device=device)
        segmentation_obj._image_size = layout.dimensions

    # Generate images and annotations and save them
    kwargs = {
        "layout": layout,
        "use_sam": use_sam,
        "segmentation_obj": segmentation_obj,
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "target_height": height,
        "target_width": width,
        "expand_pixels": expand_pixels,
    }
    Parallel(n_jobs=num_cpu)(
        delayed(partial(segment_pills, **kwargs))(image_path)
        for image_path in tqdm(images_to_segment, desc="Segmenting pills")
    )


if __name__ == "__main__":
    main()
