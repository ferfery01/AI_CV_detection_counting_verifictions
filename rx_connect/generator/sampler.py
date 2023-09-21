import random
from typing import List

import pandas as pd

from rx_connect.core.types.generator import PillMaskPaths


def _extract_pill_image_mask_paths(df: pd.DataFrame) -> List[PillMaskPaths]:
    """Extract the image and mask paths from the metadata dataframe as a list."""
    return [PillMaskPaths(image_p, mask_p) for image_p, mask_p in zip(df.Image_Path, df.Mask_Path)]


def sample_from_color_shape_by_ndc(filter_df: pd.DataFrame, pill_types: int = 1) -> List[PillMaskPaths]:
    """Uniformly samples rows from the provided DataFrame based on a single, randomly chosen color-shape
    combination. All sampled pills will share this combination.

    The function proceeds as follows:
    1. It raises errors for invalid `pill_types` values.
    2. If the 'NDC9' column is absent, it samples directly from the entire DataFrame.
    3. Otherwise, it groups the data by 'Color' and 'Shape' attributes and then selects a random
        color-shape combination.
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

    Returns:
        List[PillMaskPaths]: A list of `PillMaskPaths` named tuples representing the sampled rows.
    """
    if pill_types <= 0:
        raise ValueError(f"`pill_types` should be a positive integer, but provided {pill_types}.")

    if pill_types > len(filter_df):
        raise ValueError(
            f"Requested samples exceed possible samples. Maximum samples possible are {len(filter_df)}."
        )

    if "NDC9" not in filter_df.columns:
        pill_mask_paths_list = _extract_pill_image_mask_paths(filter_df)
        return random.sample(pill_mask_paths_list, pill_types)

    # Group by color and shape attributes
    df_grouped = filter_df.groupby(["Color", "Shape"])

    # Randomly select a particular color-shape combination
    color_shape_combination = random.choice(list(df_grouped.groups.keys()))

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
