from enum import Enum
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import pandas as pd

from rx_connect.tools.logging import setup_logger

logger = setup_logger()


class Layouts(Enum):
    """The RxImage pill images have 8 different layouts. More information about the layouts can be
    obtained at https://data.lhncbc.nlm.nih.gov/public/Pills/RxImageImageLayouts.docx
    """

    C3PI_Test = "C3PI_Test"
    C3PI_Reference = "C3PI_Reference"
    MC_C3PI_REFERENCE_SEG_V1_6 = "MC_C3PI_REFERENCE_SEG_V1.6"
    MC_CHALLENGE_V1_0 = "MC_CHALLENGE_V1.0"
    MC_COOKED_CALIBRATED_V1_2 = "MC_COOKED_CALIBRATED_V1.2"
    MC_API_NLMIMAGE_V1_3 = "MC_API_NLMIMAGE_V1.3"
    MC_API_RXNAV_V1_3 = "MC_API_RXNAV_V1.3"
    MC_SPL_IMAGE_V3_0 = "MC_SPL_SPLIMAGE_V3.0"

    @classmethod
    def members(cls) -> List[str]:
        """Returns the list of layouts."""
        return [layout.name for layout in cls]

    @property
    def dimensions(self) -> Optional[Tuple[int, int]]:
        """Returns the dimensions for the layout if available, otherwise returns None."""
        return LAYOUT_METADATA[self].image_size

    @property
    def max_pills(self) -> int:
        """Returns the maximum number of pills in an image for the layout."""
        return LAYOUT_METADATA[self].max_pills

    @classmethod
    def value_to_enum(cls, value: str) -> "Layouts":
        """Converts the string to its corresponding Layouts enum. If the string is not a valid layout,
        an error is raised.
        """
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"Invalid layout: {value} not in {cls.members()}")


class Metadata(NamedTuple):
    """Metadata for each layout."""

    image_size: Tuple[int, int]
    max_pills: int


LAYOUT_METADATA: Dict[Layouts, Metadata] = {
    Layouts.C3PI_Test: Metadata(image_size=(720, 1280), max_pills=2),
    Layouts.C3PI_Reference: Metadata(image_size=(1600, 2400), max_pills=2),
    Layouts.MC_C3PI_REFERENCE_SEG_V1_6: Metadata(image_size=(1600, 2400), max_pills=1),
    Layouts.MC_CHALLENGE_V1_0: Metadata(image_size=(1600, 2400), max_pills=1),
    Layouts.MC_COOKED_CALIBRATED_V1_2: Metadata(image_size=(1600, 2400), max_pills=2),
    Layouts.MC_API_NLMIMAGE_V1_3: Metadata(image_size=(768, 1024), max_pills=2),
    Layouts.MC_API_RXNAV_V1_3: Metadata(image_size=(768, 1024), max_pills=2),
    Layouts.MC_SPL_IMAGE_V3_0: Metadata(image_size=(768, 1024), max_pills=2),
}
"""Mapping containing the metadata for each layout. The metadata contains the image dimensions to use
during the model inference and the maximum num of pills a particular layout can have. This is used to
filter out all the erroneous bonding boxes during inference.
"""


def load_consumer_image_df_by_layout(
    data_dir: Union[str, Path], layout: Optional[Layouts] = None
) -> pd.DataFrame:
    """Loads the consumer grade images csv file and filters it based on the layout. If the
    layout is not specified, the entire dataframe is returned.
    """
    csv_file = Path(data_dir) / "consumer_grade_images.csv"

    # Load the consumer grade images csv file
    df = (
        pd.read_excel("https://data.lhncbc.nlm.nih.gov/public/Pills/directory_consumer_grade_images.xlsx")
        if not csv_file.exists()
        else pd.read_csv(csv_file)
    )

    # Drop all the duplicate rows
    df = df.drop_duplicates()

    # Filter the dataframe based on the layout
    if layout is not None:
        df = df[df.Layout == layout.value]

    # Add a column for the filename
    df["FileName"] = df.Image.map(lambda x: Path(x).name)

    return df
