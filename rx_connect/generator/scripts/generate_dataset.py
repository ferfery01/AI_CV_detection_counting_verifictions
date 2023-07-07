from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Optional, Tuple, Union
from uuid import uuid4

import click
import cv2
import numpy as np
from joblib import Parallel, delayed
from tqdm import trange

from rx_connect import SHARED_EPILL_DATA_DIR
from rx_connect.generator.annotations import create_yolo_annotations
from rx_connect.generator.io_utils import SEGMENTATION_LABELS, YOLO_LABELS
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
    img_comp, mask_comp, labels_comp = generator()
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

    # Save instance segmentation masks
    if mode in ("both", "segmentation"):
        np.save(f"{output_folder}/{mode}/{SEGMENTATION_LABELS}/{_id}.npy", mask_comp)


@click.command()
@click.option(
    "-p",
    "--pill-mask-path",
    default=SHARED_EPILL_DATA_DIR,
    show_default=True,
    help="Path to the folder with pill masks. It can be a local directory or a remote path.",
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
    type=click.Choice(["detection", "segmentation", "both"]),
    show_default=True,
    help="Flag to indicate whether to generate detection, segmentation, or both types of annotations.",
)
@click.option(
    "-np",
    "--n-pill-types",
    default=1,
    show_default=True,
    help="Number of different types of pills on each image",
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
    "-ct",
    "--color-tint",
    default=5,
    type=click.IntRange(0, 20),
    help="""Controls the aggressiveness of the color tint applied to the background.
    The higher the value, the more aggressive the color tint. The value
    should be between 0 and 20. Only used if --bg-image-path is not provided.
    """,
)
@click.option(
    "-ac/-dac",
    "--apply-color/--dont-apply-color",
    default=True,
    type=bool,
    show_default=True,
    help="""Apply color augmentations to the composed image. This is useful fot simulating
    different lighting conditions.""",
)
@click.option(
    "-an/-dan",
    "--apply-noise/--dont-apply-noise",
    default=True,
    type=bool,
    show_default=True,
    help="""Apply noise augmentations to the composed image. This is useful for simulating
    different camera conditions.""",
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
    "-c",
    "--num-cpu",
    default=cpu_count() // 2,
    show_default=True,
    help="The number of CPU cores to use. Use 1 for debugging.",
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
    color_tint: int,
    apply_color: bool,
    apply_noise: bool,
    enable_defective_pills: bool,
    enable_edge_pills: bool,
    num_cpu: int,
):
    # Create output folders
    output_path = Path(output_folder)
    img_path, label_path = create_folders(output_path, mode)

    # Initialize Generator object
    generator_obj = RxImageGenerator(
        images_dir=pill_mask_path,
        bg_dir=bg_image_path,
        image_size=(min_bg_dim, max_bg_dim),
        num_pills=(min_pills, max_pills),
        num_pills_type=n_pill_types,
        max_overlap=max_overlap,
        max_attempts=max_attempts,
        color_tint=color_tint,
        apply_color=apply_color,
        apply_noise=apply_noise,
        enable_defective_pills=enable_defective_pills,
        enable_edge_pills=enable_edge_pills,
    )

    # Generate images and annotations and save them
    Parallel(n_jobs=num_cpu)(
        delayed(generate_samples)(generator_obj, output_path, mode)
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
