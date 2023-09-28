from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from rx_connect.core.types.generator import PillMaskPaths
from rx_connect.core.utils.str_utils import convert_to_string_list
from rx_connect.generator.metadata import COLORS_LIST, SHAPES_LIST, Colors, Shapes
from rx_connect.tools.logging import setup_logger

logger = setup_logger()


def parse_file_hash(file_path: Union[str, Path, PillMaskPaths]) -> str:
    """Parse the File Hash from the file path."""
    if isinstance(file_path, PillMaskPaths):
        return file_path.image_path.stem
    elif isinstance(file_path, (str, Path)):
        return Path(file_path).stem
    else:
        raise TypeError(f"Invalid type for file_path: {type(file_path)}")


def filter_by_color_and_shape(
    df: pd.DataFrame = None,
    *,
    colors: Optional[Union[str, Sequence[str], Colors, Sequence[Colors]]] = None,
    shapes: Optional[Union[str, Sequence[str], Shapes, Sequence[Shapes]]] = None,
) -> pd.DataFrame:
    """Filter the metadata dataframe based on the color and shape. If both color and
    shape are None, then no filtering is done.

    Args:
        df: The dataframe containing the metadata.
        colors: The color of the pill to keep. If None, then all colors are kept.
        shapes: The shape of the pill to keep. If None, then all shapes are kept.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    if "Color" not in df.columns or "Shape" not in df.columns:
        if colors is not None or shapes is not None:
            logger.warning(
                "The metadata dataframe does not contain the 'Color' and 'Shape' columns. "
                "Skipping the color and shape filtering."
            )
        return df

    # Convert the colors and shapes to a list of strings
    colors = convert_to_string_list(colors, Colors)
    shapes = convert_to_string_list(shapes, Shapes)
    colors = COLORS_LIST if (colors is None or len(colors) == 0) else colors
    shapes = SHAPES_LIST if (shapes is None or len(shapes) == 0) else shapes

    # Filter the dataframe based on the color and shape
    filter_df = df[pd.Series(df.Color.isin(colors) & df.Shape.isin(shapes))]
    logger.info(f"Found {len(filter_df)} pills with:\n color: {colors}\n shape: {shapes}.")

    return filter_df
