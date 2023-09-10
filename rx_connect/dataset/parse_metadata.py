import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Union

import click
import pandas as pd
from tqdm import tqdm

from rx_connect import ROOT_DIR
from rx_connect.core.utils.str_utils import str_to_hash
from rx_connect.dataset.utils import Layouts
from rx_connect.tools.logging import setup_logger

logger = setup_logger()


def parse_xml_to_df(file_path: Union[str, Path]) -> pd.DataFrame:
    """Parse the XML file and extract data into a DataFrame."""
    # Parse the XML
    tree = ET.parse(file_path)
    root = tree.getroot()

    # List to hold the records
    records: List[Dict[str, Any]] = []

    # Iterate through each <Image> tag to extract data
    for image in root.findall(".//Image"):
        record: Dict[str, Any] = {}
        for child in image:
            # Check if the tag has child elements (like the <File> tag)
            if len(child):
                for subchild in child:
                    # Prefix the child tag's name with the parent tag's name
                    col_name = f"{child.tag}_{subchild.tag}"
                    record[col_name] = subchild.text
            else:
                record[child.tag] = child.text
        records.append(record)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(records)
    return df


def postprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Postprocess the DataFrame. This includes merging the Class and Layout columns and converting
    the File_Name column to a hash of the file name."""
    # Merge Class and Layout columns into a single column called Layout and convert
    # the values to Layout enum. The Class column is dropped.
    df.Layout = df.Layout.fillna(df.Class)
    df.Layout = df.Layout.apply(lambda x: Layouts.value_to_enum(x))
    df = df.drop(columns=["Class"])

    # Convert the File_Name column to a hash of the file name
    # The pills parsed from these images are saved using the hash as the file name
    df["File_Hash"] = df.File_Name.apply(lambda x: str_to_hash(x))

    # Clean up the Shape column by stripping off anything in parantheses and replacing spaces and
    # dashes with underscores
    df.Shape = df.Shape.str.replace(r"\([^)]*\)", "", regex=True).str.strip()
    df.Shape = df.Shape.str.replace(r"[ -]", "_", regex=True)

    # Filter out all the duplicate rows
    df = df.drop_duplicates()
    return df


@click.command()
@click.option(
    "-d",
    "--data-dir",
    default=f"{ROOT_DIR}/data/Pill_Images",
    type=click.Path(exists=True),
    help="""Path to the directory containing the pill images and the associated csv file from the
    Computational Photography Project for Pill Identification (C3PI).""",
)
def main(data_dir: Union[str, Path]) -> None:
    data_dir = Path(data_dir)
    output_file = data_dir / "metadata.csv"
    if output_file.exists():
        logger.warning(f"Existing metadata file found at {output_file} will be overwritten.")

    # Step 1: Generate a list of all the file paths
    file_paths = list((data_dir / "ALLXML").glob("*.xml"))
    if len(file_paths) == 0:
        raise ValueError(
            "No XML files found in the data directory. Did you download the data? "
            "If not, run `python rx_connect.datasets.scrape_nih_images` to download the data first."
        )

    # Step 2: Loop through each file path and parse it to a DataFrame
    dataframes = [parse_xml_to_df(file_path) for file_path in tqdm(file_paths, desc="Parsing XML files")]

    # Step 3: Concatenate all the DataFrames to create a single DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Step 4: Postprocess the DataFrame
    final_df = postprocess_df(merged_df)

    # Step 5: Save the DataFrame to a CSV file
    final_df.to_csv(output_file, index=False)
    logger.info(f"Saved metadata to {output_file}")


if __name__ == "__main__":
    main()
