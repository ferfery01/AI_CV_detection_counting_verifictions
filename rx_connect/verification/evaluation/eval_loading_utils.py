import random
from pathlib import Path
from typing import List, Tuple

from rx_connect.tools.logging import setup_logger

logger = setup_logger()


def load_pill_images_and_references(
    ref_dir: Path,
    ref_list: List[Path],
    data_dir: Path,
    true_ref: str,
    img_name: str,
    nfalse_ref: int,
    seed: int,
) -> Tuple[Path, List[Path], Path, int]:
    """
    Args:
        ref_dir: The directory containing reference pills.
        ref_list: List of Path objects representing reference pills.
        data_dir: The directory containing data for evaluations.
        true_ref: The name of the true reference pill.
        img_name: The name of the image.
        nfalse_ref: The number of randomly generated false reference pills.
        seed: The random seed for reproducibility.

    Returns:
        Tuple[Path, List[Path], Path, int]: A tuple containing:
    """

    seed = +1
    # reading the file and its reference from the csv file
    random.seed(a=seed, version=2)

    # generate a false reference pill for the pill tray
    # and make sure to exclude the reference_pills_true.
    # sample_dir_exc is a set of samples exist in
    # ref_list that do not match the the true pill samples

    sample_exc = set(ref_list) - {ref_dir / true_ref}
    sample_exc_list = list(sample_exc)
    reference_false = random.sample(sample_exc_list, k=nfalse_ref)

    false_ref = [ref_dir / rf for rf in reference_false]

    refname_true = ref_dir / true_ref
    im_path = data_dir / "images" / img_name
    return (refname_true, false_ref, im_path, seed)
