from pathlib import Path
from typing import List, Sequence, Tuple, Union


def get_matching_files_in_dir(
    dir_path: Union[str, Path], wildcard_patterns: Union[str, Sequence[str]]
) -> List[Path]:
    """Get the list of files in the directory that match a set of wildcard patterns. The wildcard patterns can be
    a single string or a list of strings. The function will not search recursively.

    Args:
        dir_path: The path to the directory to search in.
        wildcard_patterns: The wildcard patterns to match the files against. Eg. "*.txt", ["*.txt", "*.jpg"]

    Returns:
        The list of paths to the files that match the wildcard patterns.
    """
    dir_path = Path(dir_path)

    if isinstance(wildcard_patterns, str):
        wildcard_patterns = [wildcard_patterns]

    list_matching_files: List[Path] = []
    for wildcard_pattern in wildcard_patterns:
        list_matching_files.extend(dir_path.glob(wildcard_pattern))

    if len(list_matching_files) == 0:
        raise ValueError(
            f"No matching files found in the given directory."
            f"\n   Directory: {dir_path}"
            f"\n   Wildcard Patterns: {wildcard_patterns}"
        )

    return sorted(list_matching_files)


def filter_matching_pairs(image_paths: List[Path], mask_paths: List[Path]) -> Tuple[List[Path], List[Path]]:
    """Filters out all the image and mask paths where either the image or mask is missing.

    NOTE: It is assumed that the image and mask paths are in the same parent directory and have the
    same name stem.

    Args:
        image_paths (list): List of image paths (jpg).
        mask_paths (list): List of mask paths (png).

    Returns:
        filtered_image_paths (list): List of filtered image paths.
        filtered_mask_paths (list): List of filtered mask paths.
    """
    # Get the set of image and mask names
    images_hash = {image_path.stem for image_path in image_paths}
    masks_hash = {mask_path.stem for mask_path in mask_paths}

    # Get the set of corrupted image and mask names
    corrupted_hash = images_hash ^ masks_hash

    # Filter out the corrupted image and mask paths
    filtered_image_paths = [image_path for image_path in image_paths if image_path.stem not in corrupted_hash]
    filtered_mask_paths = [mask_path for mask_path in mask_paths if mask_path.stem not in corrupted_hash]

    return filtered_image_paths, filtered_mask_paths
