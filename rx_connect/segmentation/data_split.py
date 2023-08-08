from pathlib import Path
from typing import Union

import click
from tqdm import tqdm

from rx_connect.core.utils.str_utils import str_to_float
from rx_connect.tools.logging import setup_logger

logger = setup_logger()


"""
This script is used to split the image and label data into train/test/val datasets.
It first generates a path hash value that falls into the range [0, 1).
Then, based on the hash value and the provided train/test/val percentages, the data is moved to the respective sets.

Splitting Logic:
- Data with a hash value in the range:
  - [0, train_percentage) is assigned to the train set.
  - [train_percentage, train_percentage + test_percentage) is assigned to the test set.
  - [train_percentage + test_percentage, 1) is assigned to the validation set.

The resulting dataset follows the COCO segmentation folder format:

datasets/
├── test
│   ├── images
│   │   ├── img_0.png
│   │   ├── ...
│   └── labels
│       ├── img_0.txt
│       ├── ...
├── train
│   ├── images
│   │   ├── img_0.png
│   │   ├── ...
│   └── labels
│       ├── img_0.txt
│       ├── ...
|── val
│   ├── images
│   │   ├── img_0.png
│   │   ├── ...
│   └── labels
│       ├── img_0.txt
│       ├── ...
"""


def split_dataset(
    image_dir: Path, label_dir: Path, dest_dir: Path, train_percentage: float, test_percentage: float
) -> None:
    """
    Split the image and label data into train/test/val datasets based on their path hash values.

    Parameters:
        image_dir (Path): Source directory for image files.
        label_dir (Path): Source directory for label files.
        dest_dir (Path): Destination directory for processed files.
        train_percentage (float): Percentage split for the train set (e.g., 0.7 for 70%).
        test_percentage (float): Percentage split for the test set (e.g., 0.2 for 20%).

    Returns:
        None: The function does not return anything; it moves the files to the respective folders.
    """

    for image_file in tqdm(image_dir.iterdir(), desc="Split data processing now..."):
        # find corresponding labels using the base name of the image file
        label_file = label_dir / f"{image_file.stem}.txt"
        assert label_file.exists(), f"label file {label_file.name} is not in {label_dir}."

        # image_hash_value falls into [0, 1). Exclusive and inclusive is tested to work.
        image_hash_value = str_to_float(image_file.name)

        # when image_hash_value falls into [0, train_percentage)
        if image_hash_value < train_percentage:
            dest_folder = dest_dir / "train"
        # when image_hash_value falls into [train_percentage, train_percentage+test_percentage)
        elif image_hash_value < train_percentage + test_percentage:
            dest_folder = dest_dir / "test"
        # when image_hash_value falls into [train_percentage+test_percentage, 1)
        else:
            dest_folder = dest_dir / "val"

        dest_image = dest_folder / "images" / image_file.name
        dest_label = dest_folder / "labels" / label_file.name

        # symlink the image to dest folder
        dest_image.symlink_to(image_file)

        # symlink the label to dest folder
        dest_label.symlink_to(label_file)


@click.command()
@click.option("-i", "--image-dir", help="Source directory for image files", type=click.Path(exists=True))
@click.option("-l", "--label-dir", help="Source directory for label files", type=click.Path(exists=True))
@click.option("-d", "--dest-dir", help="Destination directory for processed files", type=click.Path())
@click.option("-tr", "--train-percentage", type=float, default=0.7, help="Percentage split for train set")
@click.option("-te", "--test-percentage", type=float, default=0.15, help="Percentage split for test set")
def main(
    image_dir: Union[str, Path],
    label_dir: Union[str, Path],
    dest_dir: Union[str, Path],
    train_percentage: float,
    test_percentage: float,
):
    image_dir, label_dir, dest_dir = Path(image_dir), Path(label_dir), Path(dest_dir)

    # create all folders and subfolders
    for set_name in ["train", "test", "val"]:
        for subset_name in ["images", "labels"]:
            (dest_dir / set_name / subset_name).mkdir(parents=True, exist_ok=True)

    assert (
        train_percentage + test_percentage <= 1
    ), f"Wrong partitions provided: {train_percentage}/{test_percentage}/{1-train_percentage-test_percentage}."

    # Split and move the original image and label data to train/test/val set
    split_dataset(image_dir, label_dir, dest_dir, train_percentage, test_percentage)
    logger.info(
        f"Images in {image_dir} and labels in {label_dir} are moved to the subfolder train/test/val in {dest_dir}"
    )


if __name__ == "__main__":
    main()
