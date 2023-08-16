from pathlib import Path
from typing import List, Sequence, Union


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
