import csv
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

"""This module contains the currently supported colors and shapes in the dataset generation
pipeline.
"""


class Colors(Enum):
    BLACK = auto()
    BLUE = auto()
    BROWN = auto()
    GRAY = auto()
    GREEN = auto()
    ORANGE = auto()
    PINK = auto()
    PURPLE = auto()
    RED = auto()
    TURQUOISE = auto()
    WHITE = auto()
    YELLOW = auto()


class Shapes(Enum):
    BULLET = auto()
    CAPSULE = auto()
    DIAMOND = auto()
    DOUBLE_CIRCLE = auto()
    FREEFORM = auto()
    HEXAGON = auto()
    OCTAGON = auto()
    OVAL = auto()
    PENTAGON = auto()
    RECTANGLE = auto()
    ROUND = auto()
    SEMI_CIRCLE = auto()
    SQUARE = auto()
    TEAR = auto()
    TRAPEZOID = auto()
    TRIANGLE = auto()


COLORS_LIST = [color.name.upper() for color in Colors]
SHAPES_LIST = [shape.name.upper() for shape in Shapes]


class PillMetadata(NamedTuple):
    """Named tuple for storing the pill/drug metadata."""

    ref_id: str
    drug_name: str
    ndc: int
    color: str
    shape: str
    imprint: str


def append_metadata_to_csv(
    metadata_entries: List[PillMetadata], csv_file: Path, extra_columns: Optional[Dict[str, str]] = None
) -> None:
    """Appends pill metadata to an existing or new CSV file. Optionally includes additional columns.

    Parameters:
        metadata_entries: List of PillMetadata namedtuples to append to the CSV file.
        csv_file: File path for the target CSV file.
        extra_columns: Optional dictionary containing additional column names and their uniform values.
    """
    # Initialize extra_columns if it's None
    extra_columns = extra_columns or {}

    # Check if the CSV file already exists to decide whether to write headers
    write_headers = not csv_file.exists()

    # Combine original and additional field names for the CSV
    field_names = list(extra_columns.keys()) + list(metadata_entries[0]._fields)

    # Open the CSV file in append mode
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)

        # Write headers only if the CSV file is newly created
        if write_headers:
            writer.writeheader()

        # Loop through each metadata entry to write to the CSV
        for entry in metadata_entries:
            row_data = entry._asdict()

            # Merge additional columns into the row data
            row_data.update(extra_columns)

            # Write the completed row to the CSV
            writer.writerow(row_data)
