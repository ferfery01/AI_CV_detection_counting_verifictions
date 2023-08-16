from pathlib import Path
from typing import NamedTuple, Optional, Union

import click
from tqdm import tqdm

from rx_connect.core.types.generator import COCO_LABELS, SEGMENTATION_LABELS
from rx_connect.core.utils.io_utils import get_matching_files_in_dir
from rx_connect.core.utils.str_utils import str_to_float
from rx_connect.tools.logging import setup_logger

logger = setup_logger()


"""This script is used to split the image and mask and/or label data into train/test/val datasets.
It first generates a path hash value that falls into the range [0, 1). Then, based on the hash value
and the provided train/test/val percentages, the data is moved to the respective sets.

Splitting Logic:
- Data with a hash value in the range:
  - [0, train_fraction) is assigned to the train set.
  - [train_fraction, train_fraction + test_fraction) is assigned to the test set.
  - [train_fraction + test_fraction, 1) is assigned to the validation set.

The resulting dataset follows the COCO segmentation folder format:

datasets/
    ├── test
    │   ├── images
    │   │   ├── img_0.png
    │   │   ├── ...
    │   ├── labels
    │   │   ├── img_0.txt
    │   │   ├── ...
    │   └── masks
    │       ├── img_0.npy
    │       ├── ...
    ├── train
    │   ├── images
    │   │   ├── img_0.png
    │   │   ├── ...
    │   ├── labels
    │   │   ├── img_0.txt
    │   │   ├── ...
    │   └── masks
    │       ├── img_0.npy
    │       ├── ...
    |── val
    │   ├── images
    │   │   ├── img_0.png
    │   │   ├── ...
    │   ├── labels
    │   │   ├── img_0.txt
    │   │   ├── ...
    │   └── masks
    │       ├── img_0.npy
    │       ├── ...
"""


class LabelPaths(NamedTuple):
    """Named tuple to hold the label directories."""

    mask_dir: Optional[Path] = None
    anno_dir: Optional[Path] = None


def split_dataset(
    image_dir: Path, label_dir: LabelPaths, dest_dir: Path, train_fraction: float, test_fraction: float
) -> None:
    """
    Split the image and label data into train/test/val datasets based on their path hash values.

    Parameters:
        image_dir (Path): Source directory for image files.
        label_dir (LabelPaths): Source directory for corresponding mask and COCO annotation files.
        dest_dir (Path): Destination directory for processed files.
        train_fraction (float): fraction split for the train set (e.g., 0.7 for 70%).
        test_fraction (float): fraction split for the test set (e.g., 0.2 for 20%).

    Returns:
        None: The function does not return anything; it moves the files to the respective folders.
    """
    # Get all image files in the image directory
    image_paths = get_matching_files_in_dir(image_dir, wildcard_patterns="*.[jp][pn]g")
    mask_dir, anno_dir = label_dir.mask_dir, label_dir.anno_dir

    # Iterate over all image files
    for image_file in tqdm(image_paths, desc="Splitting dataset"):
        # image_hash_value falls into [0, 1). Exclusive and inclusive is tested to work.
        image_hash_value = str_to_float(image_file.name)

        # when image_hash_value falls into [0, train_fraction)
        if image_hash_value < train_fraction:
            dest_folder = dest_dir / "train"
        # when image_hash_value falls into [train_fraction, train_fraction+test_fraction)
        elif image_hash_value < train_fraction + test_fraction:
            dest_folder = dest_dir / "test"
        # when image_hash_value falls into [train_fraction+test_fraction, 1)
        else:
            dest_folder = dest_dir / "val"

        # Get the corresponding mask and label files and symlink them to dest folder
        if mask_dir is not None:
            mask_file = mask_dir / f"{image_file.stem}.npy"
            assert mask_file.exists(), f"mask file {mask_file.name} is not in {mask_dir}."
            (dest_folder / "masks" / mask_file.name).symlink_to(mask_file.resolve())
        if anno_dir is not None:
            anno_file = anno_dir / f"{image_file.stem}.txt"
            assert anno_file.exists(), f"label file {anno_file.name} is not in {anno_dir}."
            (dest_folder / "labels" / anno_file.name).symlink_to(anno_file.resolve())

        # symlink the image to dest folder
        (dest_folder / "images" / image_file.name).symlink_to(image_file.resolve())


@click.command()
@click.option("-d", "--data-dir", help="Source directory for all the data", type=click.Path(exists=True))
@click.option("-o", "--output-dir", help="Output directory for the processed data", type=click.Path())
@click.option("-tr", "--train-fraction", type=float, default=0.7, help="Percentage split for train set")
@click.option("-te", "--test-fraction", type=float, default=0.15, help="Percentage split for test set")
def main(
    data_dir: Union[str, Path], output_dir: Union[str, Path], train_fraction: float, test_fraction: float
) -> None:
    data_dir, output_dir = Path(data_dir), Path(output_dir)

    # Initialize the images, masks, and COCO annotations directories
    image_dir = data_dir / "images"
    comp_mask_dir = data_dir / SEGMENTATION_LABELS
    coco_anno_dir = data_dir / COCO_LABELS

    # Initialize the sub-folders for the train/test/val sets
    subset_folders = ["images"]
    mask_dir, anno_dir = None, None
    if comp_mask_dir.exists():
        mask_dir = comp_mask_dir
        subset_folders.append("masks")
    if coco_anno_dir.exists():
        anno_dir = coco_anno_dir
        subset_folders.append("labels")
    assert len(subset_folders) > 1, f"At least one of `{SEGMENTATION_LABELS}` or `{COCO_LABELS}` must exist."
    label_dir = LabelPaths(mask_dir=mask_dir, anno_dir=anno_dir)

    # Create all the folders and subfolders
    for set_name in ["train", "test", "val"]:
        for folder in subset_folders:
            (output_dir / set_name / folder).mkdir(parents=True, exist_ok=True)

    assert (
        train_fraction + test_fraction < 1
    ), f"Wrong partitions provided: {train_fraction}/{test_fraction}/{1-train_fraction-test_fraction}."

    # Split and move the original image and label data to train/test/val set
    split_dataset(image_dir, label_dir, output_dir, train_fraction, test_fraction)
    logger.info(
        f"Dataset from {data_dir} directory is split into train/test/val sets in {output_dir} directory."
    )


if __name__ == "__main__":
    main()
