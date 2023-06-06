import random
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click
import cv2
import numpy as np
from joblib import Parallel, delayed
from tqdm import trange

from rx_connect.dataset_generator.composition import create_pill_comp
from rx_connect.dataset_generator.io_utils import (
    create_yolo_annotations,
    load_bg_image,
    load_pill_mask_paths,
)
from rx_connect.dataset_generator.object_overlay import generate_random_bg
from rx_connect.wbaml.utils.logging import setup_logger

logger = setup_logger()


def generate_samples(
    bg_img: Optional[np.ndarray],
    bg_img_path: Optional[Path],
    bg_img_paths: List[Path],
    output_folder: Path,
    min_bg_dim: int,
    max_bg_dim: int,
    idx,
    **kwargs,
) -> None:
    """Generate and save a single sample along with its annotation."""
    # Generate random color background if no background image is provided
    # or if a directory of background images is provided, choose a random image
    if bg_img_path is None:
        bg_img = generate_random_bg(min_bg_dim, max_bg_dim)
    elif bg_img_path.is_dir():
        bg_img = load_bg_image(random.choice(bg_img_paths), min_bg_dim, max_bg_dim)
    assert bg_img is not None, "Background image is None"

    img_comp, mask_comp, labels_comp, _ = create_pill_comp(bg_img, **kwargs)
    img_comp = cv2.cvtColor(img_comp, cv2.COLOR_RGB2BGR)

    # Save the image and annotations
    anno_yolo = create_yolo_annotations(mask_comp, labels_comp)
    n_pills: int = len(anno_yolo)
    with (output_folder / "labels" / f"{idx}_{n_pills}.txt").open("w") as f:
        for j in range(len(anno_yolo)):
            f.write(" ".join(str(el) for el in anno_yolo[j]) + "\n")
    cv2.imwrite(str(output_folder / "images" / f"{idx}_{n_pills}.jpg"), img_comp)


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
    "--bg-img-path",
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
    default=Path("./dataset/synthetic/"),
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
    "-ab/-no-ab",
    "--allow-pills-outside/--no-allow-pills-outside",
    default=True,
    show_default=True,
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
    bg_img_path: Optional[Union[str, Path]],
    output_folder: Union[str, Path],
    n_images: int,
    n_pill_types: int,
    min_pills: int,
    max_pills: int,
    max_overlap: float,
    max_attempts: int,
    min_bg_dim: int,
    max_bg_dim: int,
    allow_pills_outside: bool,
    num_cpu: int,
):
    # Convert paths to Path objects
    pill_mask_path = Path(pill_mask_path)
    bg_img_path = Path(bg_img_path) if bg_img_path is not None else None

    # Load pill mask paths
    pill_mask_paths = load_pill_mask_paths(pill_mask_path)
    logger.info(f"Found {len(pill_mask_paths)} pill masks.")

    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "labels").mkdir(parents=True, exist_ok=True)

    # Load and resize background image if provided
    bg_img: Optional[np.ndarray] = None
    bg_img_paths: List[Path] = []
    if bg_img_path is not None:
        bg_img_path = Path(bg_img_path)
        if bg_img_path.is_file():
            bg_img = load_bg_image(bg_img_path, min_bg_dim, max_bg_dim)
        else:
            bg_img_paths = list(bg_img_path.glob("*.jpg"))

    # Generate images and annotations and save them
    kwargs: Dict[str, Any] = {
        "pill_mask_paths": pill_mask_paths,
        "n_pill_types": n_pill_types,
        "min_pills": min_pills,
        "max_pills": max_pills,
        "max_overlap": max_overlap,
        "max_attempts": max_attempts,
        "allow_pill_on_border": allow_pills_outside,
    }
    Parallel(n_jobs=num_cpu)(
        delayed(generate_samples)(
            bg_img,
            bg_img_path,
            bg_img_paths,
            output_path,
            min_bg_dim,
            max_bg_dim,
            idx,
            **kwargs,
        )
        for idx in trange(n_images, desc="Generating images")
    )

    logger.info("Annotations are saved to the folder: ", output_path / "labels")
    logger.info("Images are saved to the folder: ", output_path / "images")


if __name__ == "__main__":
    main()
