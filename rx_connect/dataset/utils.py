from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from rx_connect.tools.logging import setup_logger

logger = setup_logger()

LAYOUTS: Sequence[str] = (
    "C3PI_Reference",
    "C3PI_Test",
    "MC_C3PI_REFERENCE_SEG_V1.6",
    "MC_CHALLENGE_V1.0",
    "MC_COOKED_CALIBRATED_V1.2",
    "MC_API_NLMIMAGE_V1.3",
    "MC_API_RXNAV_V1.3",
    "MC_SPL_SPLIMAGE_V3.0",
)
"""The RxImage pill images have 8 different layouts. More information about the layouts can be
obtained at https://data.lhncbc.nlm.nih.gov/public/Pills/RxImageImageLayouts.docx
"""


def load_consumer_image_df_by_layout(
    data_dir: Union[str, Path], layout: Optional[str] = None
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
        logger.assertion(layout in LAYOUTS, f"Invalid layout: {layout} not in {LAYOUTS}")
        df = df[df.Layout == layout]

    # Add a column for the filename
    df["FileName"] = df.Image.map(lambda x: Path(x).name)

    return df
