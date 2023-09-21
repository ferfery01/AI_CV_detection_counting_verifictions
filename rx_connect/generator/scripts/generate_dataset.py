from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import uuid4

import click
import cv2
from joblib import Parallel, delayed
from PIL import Image
from tqdm import trange

from rx_connect import ROOT_DIR, SHARED_RXIMAGE_DATA_DIR
from rx_connect.core.types.generator import SEGMENTATION_LABELS, YOLO_LABELS
from rx_connect.generator.annotations import create_yolo_annotations
from rx_connect.generator.metadata import COLORS_LIST, SHAPES_LIST
from rx_connect.pipelines.generator import RxImageGenerator
from rx_connect.tools.logging import setup_logger

logger = setup_logger()


def create_folders(output_path: Path, mode: str) -> Tuple[Path, List[Path]]:
    """Create the output folder and subfolders depending on the mode.

    Args:
        output_path (Union[str, Path]): The path to the output folder.
        mode (str): The mode of the dataset generation. Can be either "detection", "segmentation",
            or "both".

    Returns:
        A tuple containing the path to the images folder and a list of paths to the labels folders.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    img_path = output_path / mode / "images"
    img_path.mkdir(parents=True, exist_ok=True)

    detection_label_path = output_path / mode / YOLO_LABELS
    segmentation_label_path = output_path / mode / SEGMENTATION_LABELS

    label_paths = []
    if mode in ("both", "detection"):
        label_paths.append(detection_label_path)
    if mode in ("both", "segmentation"):
        label_paths.append(segmentation_label_path)

    for label_path in label_paths:
        label_path.mkdir(parents=True, exist_ok=True)

    return img_path, label_paths


def generate_samples(generator: RxImageGenerator, output_folder: Path, mode: str) -> None:
    """Generate and save a single sample along with its annotation."""
    # Generate the synthetic image along with its annotation
    img_comp, mask_comp, labels_comp, *_ = generator()
    img_comp = cv2.cvtColor(img_comp, cv2.COLOR_RGB2BGR)

    # Generate a unique ID for the image
    _id = uuid4()

    # Save the composed image
    cv2.imwrite(f"{output_folder}/{mode}/images/{_id}.jpg", img_comp)

    # Save YOLO annotations
    if mode in ("both", "detection"):
        anno_yolo = create_yolo_annotations(mask_comp, labels_comp)
        with (output_folder / mode / YOLO_LABELS / f"{_id}.txt").open("w") as f:
            for j in range(len(anno_yolo)):
                f.write(" ".join(str(el) for el in anno_yolo[j]) + "\n")

        # Save mapping between composed image and sampled pill paths
        with (output_folder / mode / "pill_info.csv").open("a") as f:
            f.write(f"{_id}.jpg: {', '.join(str(path) for path in generator.sampled_images_path)}\n")

    # Save instance segmentation masks as PIL images
    if mode in ("both", "segmentation"):
        mask_comp_pl = Image.fromarray(mask_comp).convert("L")
        mask_comp_pl.save(f"{output_folder}/{mode}/{SEGMENTATION_LABELS}/{_id}.png")


@click.command()
@click.option(
    "-i",
    "--input-dir",
    default=SHARED_RXIMAGE_DATA_DIR,
    type=click.Path(path_type=Path),
    show_default=True,
    help="Local or remote directory containing pill images, masks, and/or a CSV file with metadata.",
)
@click.option(
    "-b",
    "--bg-image-path",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    show_default=True,
    help="""Path to the background image. For directories, a random image is selected each iteration.
    When omitted, a random colored background is generated.""",
)
@click.option(
    "-o",
    "--output-dir",
    default=ROOT_DIR / "data" / "synthetic",
    type=click.Path(path_type=Path),
    show_default=True,
    help="Directory for saving generated outputs",
)
@click.option(
    "-m",
    "--mode",
    default="detection",
    type=click.Choice(["detection", "segmentation", "both"]),
    show_default=True,
    help="Flag to indicate whether to generate detection, segmentation, or both types of annotations.",
)
@click.option(
    "-n",
    "--n-images",
    default=100,
    show_default=True,
    help="Specify the number of images to generate.",
)
@click.option(
    "-np",
    "--n-pill-types",
    default=1,
    show_default=True,
    help="Number of different types of pills on each image",
)
@click.option(
    "-pf",
    "--pill-fractions",
    type=float,
    multiple=True,
    help="""Fractions of pills per drug type. The fractions must sum to 1. If omitted, the
    fractions are sampled uniformly.""",
)
@click.option(
    "-c",
    "--colors",
    multiple=True,
    type=click.Choice(COLORS_LIST, case_sensitive=True),
    help="Specify the color of the pills to use for generating the images.",
)
@click.option(
    "-s",
    "--shapes",
    multiple=True,
    type=click.Choice(SHAPES_LIST, case_sensitive=True),
    help="Specify the shape of the pills to use for generating the images.",
)
@click.option(
    "-sc",
    "--scale",
    nargs=2,
    type=float,
    default=[0.1, 1.0],
    show_default=True,
    help="""The range of scaling factors to apply to the pills. The scaling factor is randomly sampled
    from the range. The same scaling to both the width and height is applied. To use a fixed scaling factor,
    provide the same value for both the minimum and maximum values.
    """,
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
    default=0.1,
    show_default=True,
    help="Maximum overlap between pills",
)
@click.option(
    "-ma",
    "--max-attempts",
    default=5,
    show_default=True,
    help="Maximum number of attempts to place a pill",
)
@click.option(
    "-mb",
    "--min-bg-dim",
    default=2160,
    show_default=True,
    help="Minimum dimension of the background image",
)
@click.option(
    "-MB",
    "--max-bg-dim",
    default=3840,
    show_default=True,
    help="Maximum dimension of the background image",
)
@click.option(
    "-aba/-dba",
    "--apply-bg-aug/--dont-apply-bg-aug",
    default=True,
    show_default=True,
    help="""Whether to apply augmentations to the background image. This is useful for generating
    images with diverse backgrounds.""",
)
@click.option(
    "-aca/-dca",
    "--apply-composed-aug/--dont-apply-composed-aug",
    default=True,
    show_default=True,
    help="""Whether to apply augmentations to the composed image. This is useful for simulating
    different lighting conditions and camera conditions.""",
)
@click.option(
    "-ed",
    "--enable-defective-pills",
    is_flag=True,
    help="Allow defective pills to be placed on the background image",
)
@click.option(
    "-ep",
    "--enable-edge-pills",
    is_flag=True,
    help="Allow pills to be placed on the edge of the background image",
)
@click.option(
    "-nc",
    "--num-cpu",
    default=cpu_count() // 2,
    show_default=True,
    help="The number of CPU cores to use. Use 1 for debugging.",
)
def main(
    input_dir: Path,
    bg_image_path: Optional[Path],
    output_dir: Path,
    mode: str,
    n_images: int,
    n_pill_types: int,
    pill_fractions: Tuple[float, ...],
    colors: Tuple[str, ...],
    shapes: Tuple[str, ...],
    scale: Tuple[float, float],
    min_pills: int,
    max_pills: int,
    max_overlap: float,
    max_attempts: int,
    min_bg_dim: int,
    max_bg_dim: int,
    apply_bg_aug: bool,
    apply_composed_aug: bool,
    enable_defective_pills: bool,
    enable_edge_pills: bool,
    num_cpu: int,
) -> None:
    # Create output folders
    img_path, label_path = create_folders(output_dir, mode)

    # Initialize Generator object
    generator_obj = RxImageGenerator(
        data_dir=input_dir,
        bg_dir=bg_image_path,
        image_size=(min_bg_dim, max_bg_dim),
        num_pills=(min_pills, max_pills),
        num_pills_type=n_pill_types,
        fraction_pills_type=pill_fractions if len(pill_fractions) > 0 else None,
        colors=colors,
        shapes=shapes,
        scale=scale,
        max_overlap=max_overlap,
        max_attempts=max_attempts,
        apply_bg_aug=apply_bg_aug,
        apply_composed_aug=apply_composed_aug,
        enable_defective_pills=enable_defective_pills,
        enable_edge_pills=enable_edge_pills,
    )

    # Generate images and annotations and save them
    Parallel(n_jobs=num_cpu)(
        delayed(generate_samples)(generator_obj, output_dir, mode)
        for _ in trange(n_images, desc="Generating images")
    )

    # Log output folder path
    msgs = []
    if mode in ("both", "detection"):
        msgs.append(f"Annotations are saved to the folder: {label_path[0]}")
    if mode == "segmentation":
        msgs.append(f"Instance segmentation masks are saved to the folder: {label_path[0]}")
    elif mode == "both":
        msgs.append(f"Instance segmentation masks are saved to the folder: {label_path[1]}")

    logger.info("\n".join(msgs))
    logger.info(f"Images are saved to the folder: {img_path}")


if __name__ == "__main__":
    main()
