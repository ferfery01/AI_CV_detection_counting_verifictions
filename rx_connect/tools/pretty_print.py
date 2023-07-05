from typing import Any, Dict, Sequence, Union

import torch
from rich.console import Console
from rich.table import Table

__all__: Sequence[str] = ("print_dict_as_table", "print_dict_as_table_transposed", "print_dict")


def print_dict_as_table_transposed(
    d: Dict[str, Union[str, float, torch.Tensor]],
    key_col: str = "Key",
    value_col: str = "Value",
    header_style: str = "bold magenta",
) -> None:
    """Pretty print a dictionary as a table. This is useful when the dictionary is
    just a single row of data. In this case, the table is transposed so that the
    keys are in the first column and the values are in the second column.

    Args:
        d (Dict[str, Union[str, float, torch.Tensor]]): Dictionary to print.
        key_col (str): Name of the key column.
        value_col (str): Name of the value column.
        header_style (str, optional): Style of the header. Defaults to "bold magenta".

    For instance, for the following dictionary:
        d = {
            'loss': 0.1234,
            'accuracy': 0.9876,
            'precision': 0.8765,
        }
        the output of `print_dict_as_table_transposed(d, 'loss', 'value')` will be:
            ┏━━━━━━━━━━━┳━━━━━━━━━━━━━┓
            ┃ loss      ┃ value       ┃
            ┡━━━━━━━━━━━╇━━━━━━━━━━━━━┩
            │ loss      │ 0.1234      │
            │ accuracy  │ 0.9876      │
            │ precision │ 0.8765      │
            └───────────┴─────────────┘
    """
    # Initialize a console
    console = Console()

    # Create a table
    table = Table(show_header=True, header_style=header_style)

    # Add columns names for the key and value columns
    table.add_column(key_col)
    table.add_column(value_col)

    # Add rows
    for key, value in d.items():
        value = f"{value:.4f}" if isinstance(value, (float, torch.Tensor)) else value
        table.add_row(key, f"{value}")

    # Print the table to the console
    console.print(table)


def print_dict_as_table(d: Dict[str, Any], header_style: str = "bold magenta") -> None:
    """Pretty print a dictionary as a table.

    For instance, the following dictionary:
        d = {
            'Name': ['John', 'Anna', 'Peter'],
            'Age': [25, 24, 33],
            'City': ['New York', 'Los Angeles', 'Chicago']
        }
        will be printed as:
                ┏━━━━━━━┳━━━━━┳━━━━━━━━━━━━━┓
                ┃ Name  ┃ Age ┃ City        ┃
                ┡━━━━━━━╇━━━━━╇━━━━━━━━━━━━━┩
                │ John  │ 25  │ New York    │
                │ Anna  │ 24  │ Los Angeles │
                │ Peter │ 33  │ Chicago     │
                └───────┴─────┴─────────────┘
    """
    # Initialize a console
    console = Console()

    # Create a table
    table = Table(show_header=True, header_style=header_style)

    # Add columns
    for column_name in d.keys():
        table.add_column(column_name)

    # Add rows
    for row in zip(*[v if isinstance(v, list) else [v] for v in d.values()]):
        table.add_row(*map(str, row))

    # Print the table to the console
    console.print(table)


def any_value_is_list(my_dict: Dict[str, Any]) -> bool:
    """Check if any value in a dictionary is a list."""
    return any(isinstance(value, list) for value in my_dict.values())


def print_dict(d: dict, header_style: str = "bold magenta", **kwargs: str) -> None:
    """Pretty print a dictionary. If any value in the dictionary is a list, then
    the dictionary is printed as a table. Otherwise, the dictionary is printed as a
    transposed table.
    """
    if any_value_is_list(d):
        print_dict_as_table(d, header_style)
    else:
        print_dict_as_table_transposed(d, header_style=header_style, **kwargs)
