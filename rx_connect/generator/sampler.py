import random
from functools import partial
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from rx_connect.core.types.generator import PillMaskPaths


def _extract_pill_image_mask_paths(df: pd.DataFrame) -> List[PillMaskPaths]:
    """Extract the image and mask paths from the metadata dataframe as a list."""
    return [PillMaskPaths(image_p, mask_p) for image_p, mask_p in zip(df.Image_Path, df.Mask_Path)]


def _weighted_sampling(d: Dict[str, Union[int, float]], use_inverse_weights: bool) -> str:
    """Sample a key based on weighted probabilities.

    Args:
        d: Dictionary containing keys and their counts
        use_inverse_weights: Whether to use inverse counts for hard sampling or actual
            counts for random sampling

    Returns:
        Selected key as per the sampling strategy
    """
    # Get weights based on the flag
    weights = {k: 1 / v if v > 0 else 0 for k, v in d.items()} if use_inverse_weights else d

    # Normalize probabilities
    total = sum(weights.values())
    normalized_probs = {k: v / total for k, v in weights.items()}

    # Sample based on weighted probabilities
    return random.choices(list(normalized_probs.keys()), list(normalized_probs.values()), k=1)[0]


def _uniform_sampling(d: Dict[str, int]) -> str:
    """Sample a key where all keys have equal probability."""
    return random.choice(list(d.keys()))


SAMPLING_METHODS_MAP: Dict[str, Callable[[Dict[str, int]], str]] = {
    "uniform": _uniform_sampling,
    "random": partial(_weighted_sampling, use_inverse_weights=False),
    "hard": partial(_weighted_sampling, use_inverse_weights=True),
}
"""Mapping of sampling methods to the corresponding sampling functions.
"""


def sample_from_color_shape_by_ndc(
    filter_df: pd.DataFrame, pill_types: int = 1, sampling: str = "uniform"
) -> List[PillMaskPaths]:
    """Samples rows from the provided DataFrame based on a color-shape combination selected by the
    provided sampling method. All sampled pills will share this combination unless the DataFrame does
    not have enough rows with the selected combination.

    The function proceeds as follows:
    1. It raises errors for invalid `pill_types` values.
    2. If the 'NDC9' column is absent, it samples directly from the entire DataFrame.
    3. Otherwise, it groups the data by 'Color' and 'Shape' attributes and then selects a particular
        color-shape combination based on the provided `sampling` method.
    4. From this combination, it picks unique 'NDC9' codes.
    5. If the number of unique 'NDC9' codes from the chosen combination is less than the required
        `pill_types`, the function supplements the selection by randomly sampling additional rows
        from the remaining data.
    6. The final list of selected rows is converted to a list of `PillMaskPaths` named tuples for the
        return value.

    Args:
        filter_df (pd.DataFrame): DataFrame with required columns "NDC9", "Color", "Shape",
            "Image_Path", and "Mask_Path".
        pill_types (int): Desired number of different pill types to sample. Defaults to 1.
        sampling (str): Sampling method to use. Can be one of "uniform", "random", or "hard". Defaults
            to "uniform".

    Returns:
        List[PillMaskPaths]: A list of `PillMaskPaths` named tuples representing the sampled rows.
    """
    if pill_types <= 0:
        raise ValueError(f"`pill_types` should be a positive integer, but provided {pill_types}.")

    if pill_types > len(filter_df):
        raise ValueError(
            f"Requested samples exceed possible samples. Maximum samples possible are {len(filter_df)}."
        )

    if sampling not in SAMPLING_METHODS_MAP:
        raise ValueError(
            f"Sampling method should be one of {list(SAMPLING_METHODS_MAP.keys())}, but provided {sampling}."
        )

    if "NDC9" not in filter_df.columns:
        pill_mask_paths_list = _extract_pill_image_mask_paths(filter_df)
        return random.sample(pill_mask_paths_list, pill_types)

    # Group by color and shape attributes
    df_grouped = filter_df.groupby(["Color", "Shape"])

    # Select a particular color-shape combination based on the sampling method
    combined_count_dict = df_grouped.size().to_dict()
    color_shape_combination = SAMPLING_METHODS_MAP[sampling](combined_count_dict)

    # Get all rows with the selected color-shape combination
    grouped_samples = df_grouped.get_group(color_shape_combination).sample(frac=1)

    # Remove duplicates from the original dataframe
    unique_samples = grouped_samples.drop_duplicates(subset="NDC9")

    # Remove the unique samples from the original dataframe
    all_rem_samples = filter_df[~filter_df.index.isin(unique_samples.index)]

    # Determine additional samples needed
    additional_samples_needed = pill_types - len(unique_samples)

    # If additional samples are needed, sample randomly from the duplicate rows
    if additional_samples_needed > 0:
        random_samples = all_rem_samples.sample(
            additional_samples_needed, replace=len(all_rem_samples) < additional_samples_needed
        )
        final_samples = pd.concat([unique_samples, random_samples])
    else:
        final_samples = unique_samples.sample(pill_types)

    # Convert the sampled rows to a list of PillMaskPaths NamedTuples
    pill_mask_paths_list = _extract_pill_image_mask_paths(final_samples)

    return pill_mask_paths_list


def sample_pill_location(pill_size: Tuple[int, int], bg_size: Tuple[int, int]) -> Tuple[int, int]:
    """Sample the top left corner of the pill to be placed on the background image.

    The position is sampled from a normal distribution with mean at the center of the background image.
    The standard deviation is a quarter of the background image's width and height.
    """
    # Get the height and width of the pill and background image.
    h_bg, w_bg = bg_size
    h_pill, w_pill = pill_size

    # Get the effective height and width where the pill can be placed.
    H_eff, W_eff = h_bg - h_pill, w_bg - w_pill

    # Sample the center of the pill from a normal distribution.
    x, y = np.random.normal(
        loc=(W_eff / 2, H_eff / 2), scale=(abs(W_eff / 4), abs(H_eff / 4)), size=(2,)
    ).astype(int)

    # Clip the location to ensure that the pill is within the background image.
    top_left = np.clip(x, -w_pill, w_bg), np.clip(y, -h_pill, h_bg)

    return top_left
