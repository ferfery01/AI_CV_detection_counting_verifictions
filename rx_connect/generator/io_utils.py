import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import albumentations as A
import numpy as np
import pandas as pd
from skimage.io import imread

from rx_connect import CACHE_DIR
from rx_connect.core.types.generator import SEGMENTATION_LABELS, PillMask, PillMaskPaths
from rx_connect.core.utils.io_utils import get_matching_files_in_dir
from rx_connect.generator.metadata_filter import parse_file_hash
from rx_connect.generator.transform import BACKGROUND_TRANSFORMS, resize_bg
from rx_connect.tools import is_remote_dir
from rx_connect.tools.data_tools import (
    fetch_file_paths_from_remote_dir,
    fetch_from_remote,
)
from rx_connect.tools.logging import setup_logger

logger = setup_logger()


def get_unmasked_image_paths(image_folder: Union[str, Path], output_folder: Union[str, Path]) -> List[Path]:
    """Get the paths of the images in the source folder that have not been masked yet.

    This function compares the source image folder and the output folder, and identifies
    images in the source folder that do not have corresponding masks in the output folder.

    Args:
        image_folder (Union[str, Path]): The path to the source image folder.
        output_folder (Union[str, Path]): The path to the folder where masked images are saved.

    Returns:
        List[Path]: List of Paths of the images in the source folder that have not been masked yet.
    """
    # Convert to pathlib.Path objects for easy and consistent path manipulations
    image_dir, output_dir = Path(image_folder), Path(output_folder)

    # Create the output folder if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather the names of all images in the source folder
    source_images: Set[str] = {img_path.name for img_path in image_dir.glob("*.jpg")}

    # Gather the names of all images in the output folder (assumed to be masked images)
    masked_images: Set[str] = {mask_path.name for mask_path in output_dir.glob("*.jpg")}

    # Identify images that have not been masked yet by finding the difference
    unmasked_images: Set[str] = source_images.difference(masked_images)

    # Convert the names of unmasked images to full path for further processing
    unmasked_image_paths: List[Path] = [image_dir / img_name for img_name in unmasked_images]

    logger.info(f"Found {len(unmasked_image_paths)} images to mask.")

    return unmasked_image_paths


def load_metadata(data_dir: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Loads metadata for pill images and masks from a specified directory. The metadata
    is stored in a CSV file named `metadata.csv`.

    Args:
        data_dir: The directory containing all the pill images, the corresponding
            masks, and/or the associated metadata. It can be a local directory or a
            remote directory on AI Lab GPU servers.

    Returns:
        Optional[pd.DataFrame]: DataFrame containing metadata associated with all the pill images.
            The index of the DataFrame is the file hash of the pill image. If the metadata file
            is not found, then None is returned.
    """
    data_dir = Path(data_dir)
    df_path = data_dir / "metadata.csv"

    try:
        df_path = fetch_from_remote(df_path, cache_dir=CACHE_DIR / data_dir.name)
        df = pd.read_csv(df_path)
        return df
    except FileNotFoundError:
        logger.warning(f"Metadata file not found at {df_path}. Skipping loading metadata.")
    except Exception as e:
        logger.error(f"Error loading metadata from {df_path}: {e}")

    return None


def enrich_metadata_with_paths(
    metadata_df: Optional[pd.DataFrame], file_hash_to_paths: Dict[str, PillMaskPaths]
) -> pd.DataFrame:
    """Enriches the provided metadata DataFrame with image and mask paths based on file hash keys.

    If the metadata DataFrame is not provided, a new DataFrame is created using the
    file_hash_to_paths dictionary. The function will then map file hashes to their
    corresponding image and mask paths and return the enriched or newly created DataFrame.

    Args:
        metadata_df (Optional[pd.DataFrame]): DataFrame containing metadata associated
            with pill images, indexed by file hash. If not provided, a new DataFrame is
            created from the file_hash_to_paths dictionary.
        file_hash_to_paths (Dict[str, PillMaskPaths]): Dictionary mapping file hashes
            to their respective PillMaskPaths namedtuples.

    Returns:
        pd.DataFrame: DataFrame with "Image_Path" and "Mask_Path" columns added or updated.
    """
    # Convert the dictionary to a DataFrame
    paths_df = pd.DataFrame.from_dict(file_hash_to_paths, orient="index").reset_index()
    paths_df.columns = ["File_Hash", "Image_Path", "Mask_Path"]

    # Merge with the provided metadata DataFrame or use the paths DataFrame if metadata is absent
    merged_df = paths_df if metadata_df is None else paths_df.merge(metadata_df, on="File_Hash", how="left")
    merged_df = merged_df.set_index("File_Hash")

    return merged_df


def load_pill_mask_paths(data_dir: Union[str, Path]) -> Dict[str, PillMaskPaths]:
    """Load all the pill images and the corresponding masks path.

    Args:
        data_dir: The directory containing all the pill images and the corresponding
            masks. The directory should have two subdirectories: "images" and "masks".
            It can be a local directory or a remote directory on AI Lab GPU servers.

    Returns:
        pill_mask_paths: The paths to the pill image and mask.
    """
    data_dir = Path(data_dir)

    if is_remote_dir(data_dir):
        imgs_path = fetch_file_paths_from_remote_dir(data_dir / "images")
        masks_path = fetch_file_paths_from_remote_dir(data_dir / "masks")
    else:
        imgs_path = get_matching_files_in_dir(data_dir / "images", "*.jpg")
        masks_path = get_matching_files_in_dir(data_dir / "masks", "*.[jp][pn]g")
        """New masks are saved as PNG format as it can handle binary data without loss of information.
        However, old masks are saved as JPG format. Hence, we need to check for both formats.
        """

    # Sort the file paths to ensure that the images and masks are aligned
    imgs_path, masks_path = sorted(imgs_path), sorted(masks_path)

    if len(imgs_path) == 0:
        raise FileNotFoundError(f"Could not find any pill images in {data_dir}/images.")

    if len(imgs_path) != len(masks_path):
        raise ValueError(f"Number of images ({len(imgs_path)}) and masks ({len(masks_path)}) do not match.")

    hash_to_img_mask_paths = {
        parse_file_hash(img_path): PillMaskPaths(img_path, mask_path)
        for img_path, mask_path in zip(imgs_path, masks_path)
    }
    logger.info(f"Found {len(hash_to_img_mask_paths)} pill images and masks in {data_dir}.")

    return hash_to_img_mask_paths


def load_comp_mask_paths(data_dir: Union[str, Path]) -> List[Path]:
    """Load all the masks path.

    Args:
        data_dir: The directory containing all the pill images and the corresponding
            masks. The directory should have two subdirectories: "images" and "masks".
            It can be a local directory or a remote directory on AI Lab GPU servers.

    Returns:
        pill_mask_paths: The paths to the pill image and mask.
    """
    data_dir = Path(data_dir)

    if is_remote_dir(data_dir):
        masks_path = fetch_file_paths_from_remote_dir(data_dir / SEGMENTATION_LABELS)
    else:
        masks_path = get_matching_files_in_dir(data_dir / SEGMENTATION_LABELS, "*.png")

    # Sort the file paths to ensure that the images and masks are aligned
    masks_path = sorted(masks_path)

    if len(masks_path) == 0:
        raise FileNotFoundError(f"Could not find any masks in {data_dir}/{SEGMENTATION_LABELS}.")

    logger.info(f"Found {len(masks_path)} masks.")

    return masks_path


def load_image_and_mask(
    image_path: Union[str, Path], mask_path: Union[str, Path], thresh: int = 25
) -> PillMask:
    """Get the image and boolean mask from the pill mask paths.

    Args:
        image: The path to the pill image. Can be a local path or a remote path.
        mask: The path to the pill mask. Can be a local path or a remote path.
        thresh: The threshold at which to binarize the mask. Useful only for the old ePillID masks.

    Returns:
        img: The pill image as a numpy array.
        mask: The pill mask as a boolean array.
    """
    # Fetch the image and mask from remote server, if necessary
    image_path = fetch_from_remote(image_path, cache_dir=CACHE_DIR / "images", skip_check=True)
    mask_path = fetch_from_remote(mask_path, cache_dir=CACHE_DIR / "masks", skip_check=True)

    # Load the pill image
    image = imread(image_path)

    # Load the pill mask
    mask = imread(mask_path, as_gray=True)

    # Binarize the mask if it is not binary already
    # The new masks contain only 0s and 1s, but the old masks can be anything between 0 and 255
    if len(np.unique(mask)) > 2:
        mask[mask <= thresh] = 0
        mask[mask > thresh] = 1

    return PillMask(image=image, mask=mask.astype(bool))


def load_pills_and_masks(
    image_mask_path: Sequence[PillMaskPaths], *, thresh: int = 25
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load all the pill images and the corresponding masks provided in the paths.

    Args:
        image_mask_path: The paths to the pill images and masks.
        thresh: The threshold at which to binarize the mask.

    Returns:
        pill_images: The pill images.
        pill_masks: The pill masks.
    """
    if len(image_mask_path) == 0:
        raise ValueError("`image_mask_path` is empty")

    pill_images: List[np.ndarray] = []
    pill_masks: List[np.ndarray] = []

    for img_path, mask_path in image_mask_path:
        pill_img, pill_mask = load_image_and_mask(img_path, mask_path, thresh=thresh)

        pill_images.append(pill_img)
        pill_masks.append(pill_mask)

    return pill_images, pill_masks


def load_random_pills_and_masks(
    image_mask_paths: Sequence[PillMaskPaths], *, pill_types: int = 1, thresh: int = 25
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load `pill_types` random pills and masks from the given paths.

    Args:
        image_mask_paths: The paths to the pill images and masks.
        pill_types: The number of pills to sample.
        thresh: The threshold at which to binarize the mask.

    Returns:
        Tuple of lists of all pill images and masks.
    """
    pill_images: List[np.ndarray] = []
    pill_masks: List[np.ndarray] = []

    # Randomly sample `pill_types` pills
    for _ in range(pill_types):
        # Randomly sample a pill image and mask
        idx = np.random.randint(len(image_mask_paths))
        image_path, mask_path = image_mask_paths[idx]

        pill_img, pill_mask = load_image_and_mask(image_path, mask_path, thresh=thresh)
        pill_images.append(pill_img)
        pill_masks.append(pill_mask)

    return pill_images, pill_masks


def load_bg_image(path: Path, min_dim: int = 1024, max_dim: int = 1920) -> np.ndarray:
    """Load and resize the background image.

    Args:
        path: The path to the background image.
        min_dim: The minimum dimension of the background image.
        max_dim: The maximum dimension of the background image.

    Returns:
        bg_img: The background image as a numpy array.
    """
    # Load the background image
    bg_img = imread(path)

    # Resize the background image
    bg_img = resize_bg(bg_img, max_dim, min_dim)

    return bg_img


def generate_random_bg(height: int, width: int) -> np.ndarray:
    """Generate a random background image.

    Args:
        height (int): height of the background image.
        width (int): width of the background image.

    Returns:
        np.ndarray: random background image.
    """
    # Generate a random color for the background
    background_color = np.random.randint(150, 256, size=(3,)).tolist()

    # Create a black background image
    background_image = np.zeros((height, width, 3), np.uint8)

    # Fill the background image with the random color
    background_image[:] = background_color

    return background_image


def get_background_image(
    path: Optional[Union[str, Path]] = None,
    min_bg_dim: int = 2160,
    max_bg_dim: int = 3840,
    apply_augmentations: bool = True,
) -> np.ndarray:
    """Get the background image.
        1. If no background image path is provided, generate a random color background
        2. If a background image path is provided, load it directly
        3. If a directory of background images is provided, choose a random image

    Args:
        path: Path to the background image or directory of background images
        min_bg_dim: Minimum dimension of the background image
        max_bg_dim: Maximum dimension of the background image
        apply_augmentations: Whether to apply augmentations to the background image

    Returns:
        Background image as a numpy array resized to the specified dimensions
    """
    if path is None:
        # Generate random color background if no background image is provided
        bg_image = generate_random_bg(min_bg_dim, max_bg_dim)
    else:
        path = Path(path)
        if path.is_file():
            # If a background image path is provided, load it
            bg_image = load_bg_image(path, min_bg_dim, max_bg_dim)
        elif path.is_dir():
            # If a directory of background images is provided, choose a random image
            paths = list(path.glob("*.jpg"))
            bg_image = load_bg_image(random.choice(paths), min_bg_dim, max_bg_dim)
        else:
            raise FileNotFoundError(f"Could not find background image file or directory at {path}.")

    if apply_augmentations:
        transforms = A.OneOf(BACKGROUND_TRANSFORMS)
        bg_image = transforms(image=bg_image)["image"]

    return bg_image
