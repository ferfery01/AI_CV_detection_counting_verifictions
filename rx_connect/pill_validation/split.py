from pathlib import Path
from typing import Union

import pandas as pd


def default_split(
    all_imgs_csv: Union[str, Path],
    val_imgs_csv: Union[str, Path],
    test_imgs_csv: Union[str, Path],
) -> pd.DataFrame:
    """Split the data into train, val, and test sets."""
    df = pd.read_csv(all_imgs_csv)
    val_df = pd.read_csv(val_imgs_csv)
    test_df = pd.read_csv(test_imgs_csv)

    df["split"] = "train"
    df.loc[df.images.isin(val_df.images), "split"] = "val"
    df.loc[df.images.isin(test_df.images), "split"] = "test"

    return df
