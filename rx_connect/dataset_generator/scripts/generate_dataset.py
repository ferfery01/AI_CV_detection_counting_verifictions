from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import cv2
from joblib import Parallel, delayed
from tqdm import trange

from rx_connect.dataset_generator.annotations import create_yolo_annotations
from rx_connect.dataset_generator.composition import generate_image
from rx_connect.dataset_generator.io_utils import (
    get_background_image,
    load_pill_mask_paths,
    load_random_pills_and_masks,
)
from rx_connect.tools.logging import setup_logger

logger = setup_logger()

_YOLO_LABELS = "labels"
_SEGMENTATION_LABELS = "comp_masks"


def create_folders(output_path: Path, mode: str) -> Tuple[Path, Path]:
    """Create the output folder and subfolders depending on the mode.

    Args:
        output_path (Union[str, Path]): The path to the output folder.
        mode (str): The mode of the dataset generation. Can be either "detection" or "segmentation".
    """
    output_path.mkdir(parents=True, exist_ok=True)

    img_path = output_path / mode / "images"
    img_path.mkdir(parents=True, exist_ok=True)

    label_path = output_path / mode / f"{_YOLO_LABELS if mode == 'detection' else _SEGMENTATION_LABELS}"
    label_path.mkdir(parents=True, exist_ok=True)

    return img_path, label_path


def generate_samples(
    bg_img_path: Optional[Union[str, Path]],
    images_path: List[Path],
    masks_path: List[Path],
    output_folder: Path,
    min_bg_dim: int,
    max_bg_dim: int,
    num_pills_type: int,
    idx: int,
    mode: str,
    **kwargs,
) -> None:
    """Generate and save a single sample along with its annotation."""
    # First, generate a background image
    bg_image = get_background_image(bg_img_path, min_bg_dim, max_bg_dim)

    # Second, get the pill images and masks to compose on the background
    pill_images, pill_masks = load_random_pills_and_masks(
        images_path, masks_path, pill_types=num_pills_type, thresh=25
    )

    # Third, generate the composed image (i.e. pills on background)
    img_comp, mask_comp, labels_comp, _ = generate_image(bg_image, pill_images, pill_masks, mode, **kwargs)
    img_comp = cv2.cvtColor(img_comp, cv2.COLOR_RGB2BGR)

    if mode == "detection":
        # Save YOLO annotations
        # print("This is the detection dataset generation mode")
        anno_yolo = create_yolo_annotations(mask_comp, labels_comp)
        n_pills: int = len(anno_yolo)
        with (output_folder / mode / _YOLO_LABELS / f"{idx}_{n_pills}.txt").open("w") as f:
            for j in range(len(anno_yolo)):
                f.write(" ".join(str(el) for el in anno_yolo[j]) + "\n")

        # save images
        cv2.imwrite(str(output_folder / mode / "images" / f"{idx}_{n_pills}.jpg"), img_comp)
    elif mode == "segmentation":
        n_pills = len(labels_comp)

        # Save instance segmentation masks images and masks
        cv2.imwrite(str(output_folder / mode / "images" / f"{idx}_{n_pills}.jpg"), img_comp)
        cv2.imwrite(str(output_folder / mode / _SEGMENTATION_LABELS / f"{idx}_{n_pills}.jpg"), mask_comp)
    else:
        raise ValueError(f"Unknown mode: {mode}")


@click.command()
@click.option(
    "-p",
    "--pill-mask-path",
    default=Path("./data/pills/"),
    type=click.Path(exists=True),
    show_default=True,
    help="Path to the folder with pill masks",
)
@click.option(
    "-b",
    "--bg-image-path",
    default=None,
    type=click.Path(exists=True),
    show_default=True,
    help="""
    Path to the background image. If directory, a random image will be chosen in each
    iteration. If not provided, a random color background will be generated.
    """,
)
@click.option(
    "-o",
    "--output-folder",
    default=Path("./data/synthetic/"),
    type=click.Path(),
    show_default=True,
    help="Path to the output folder",
)
@click.option(
    "-n",
    "--n-images",
    default=100,
    show_default=True,
    help="Number of images to generate",
)
@click.option(
    "-m",
    "--mode",
    default="detection",
    type=click.Choice(["detection", "segmentation"]),
    show_default=True,
    help="Flag for detection or segmentation dataset generation",
)
@click.option(
    "-np",
    "--n-pill-types",
    default=1,
    show_default=True,
    help="Number of different types of pills",
)
@click.option(
    "-mp",
    "--min-pills",
    default=5,
    show_default=True,
    help="Minimum number of pills per image",
)
@click.option(
    "-MP",
    "--max-pills",
    default=50,
    show_default=True,
    help="Maximum number of pills per image",
)
@click.option(
    "-mo",
    "--max-overlap",
    default=0.2,
    show_default=True,
    help="Maximum overlap between pills",
)
@click.option(
    "-ma",
    "--max-attempts",
    default=10,
    show_default=True,
    help="Maximum number of attempts to place a pill",
)
@click.option(
    "-mb",
    "--min-bg-dim",
    default=1080,
    show_default=True,
    help="Minimum dimension of the background image",
)
@click.option(
    "-MB",
    "--max-bg-dim",
    default=1920,
    show_default=True,
    help="Maximum dimension of the background image",
)
@click.option(
    "-e",
    "--enable-edge-pills",
    is_flag=True,
    help="Allow pills to be placed on the edge of the background image",
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
    bg_image_path: Optional[Union[str, Path]],
    output_folder: Union[str, Path],
    n_images: int,
    mode: str,
    n_pill_types: int,
    min_pills: int,
    max_pills: int,
    max_overlap: float,
    max_attempts: int,
    min_bg_dim: int,
    max_bg_dim: int,
    enable_edge_pills: bool,
    num_cpu: int,
):
    # Load pill mask paths
    images_path, masks_path = load_pill_mask_paths(pill_mask_path)

    # Create output folders
    output_path = Path(output_folder)
    img_path, label_path = create_folders(output_path, mode)

    # Generate images and annotations and save them
    kwargs: Dict[str, Any] = {
        "min_pills": min_pills,
        "max_pills": max_pills,
        "max_overlap": max_overlap,
        "max_attempts": max_attempts,
        "enable_edge_pills": enable_edge_pills,
    }
    Parallel(n_jobs=num_cpu)(
        delayed(generate_samples)(
            bg_image_path,
            images_path,
            masks_path,
            output_path,
            min_bg_dim,
            max_bg_dim,
            n_pill_types,
            idx,
            mode,
            **kwargs,
        )
        for idx in trange(n_images, desc="Generating images")
    )

    # Log output folder path
    if mode == "detection":
        logger.info("Annotations are saved to the folder: ", label_path)
    elif mode == "segmentation":
        logger.info("Instance segmentation masks are saved to the folder: ", label_path)
    logger.info("Images are saved to the folder: ", img_path)


if __name__ == "__main__":
    main()
