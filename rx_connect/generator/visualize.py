from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rx_connect.tools.logging import setup_logger

logger = setup_logger()


COLOR_MAPPING: Dict[str, str] = {
    "BLACK": "#424242",
    "BLUE": "#1565C0",
    "BROWN": "#A1887F",
    "GRAY": "#BDBDBD",
    "GREEN": "#81C784",
    "ORANGE": "#FFB74D",
    "PINK": "#F06292",
    "PURPLE": "#BA68C8",
    "RED": "#E57373",
    "TURQUOISE": "#4DB6AC",
    "WHITE": "#FFFFFF",
    "YELLOW": "#FFF176",
}
"""Mapping of color names to their hex codes.
"""


def generate_heatmap(
    df: pd.DataFrame, cmap: str = "YlGnBu", save_path: Optional[Union[str, Path]] = None
) -> None:
    """Generate a heatmap based on the provided data.

    Args:
        df (DataFrame): The DataFrame containing the data.
        cmap (str): The colormap to use for the heatmap.
        save_path (str or Path): The path to save the plot to. If None, the plot will be displayed.
    """
    # Create a pivot table for the heatmap
    heatmap_data = df.groupby(["color", "shape"]).size().reset_index(name="count")
    heatmap_data_pivot = heatmap_data.pivot(index="color", columns="shape", values="count").fillna(0)

    if len(heatmap_data_pivot) == 0:
        logger.warning("The DataFrame does not contain any color and shape data to generate a heatmap.")
        return None

    # Create the heatmap
    plt.figure(figsize=(20, 15))
    sns.heatmap(heatmap_data_pivot, annot=True, fmt=".0f", cmap=cmap, linewidths=0.5)
    plt.title("Heatmap of Images Based on Color and Shape")
    plt.ylabel("Color")

    # If save_path is specified, save the plot, otherwise display it
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        # Show the plot
        plt.show()


def generate_stacked_bar_chart(df: pd.DataFrame, save_path: Optional[Union[str, Path]] = None) -> None:
    """Generate a stacked bar chart with glossy colors and fixed legend.

    Args:
        df (DataFrame): The DataFrame containing the data.
        save_path (str or Path): The path to save the plot to. If None, the plot will be displayed.
    """
    if "color" not in df.columns or "shape" not in df.columns:
        logger.warning(
            "The DataFrame does not contain 'color' and 'shape' columns to generate a stacked bar chart."
        )
        return None

    # Prepare data for Stacked Bar Chart
    stacked_data = df.groupby(["shape", "color"]).size().unstack(fill_value=0).sort_values(by="shape")
    stacked_data.columns.name = None

    if len(stacked_data) == 0:
        logger.warning(
            "The DataFrame does not contain any color and shape data to generate a stacked bar chart."
        )
        return None

    # Get unique colors and shapes
    unique_colors = df["color"].unique()
    unique_shapes = df["shape"].unique()

    # Sort columns based on the color mapping to match the legend
    sorted_colors = [color for color in COLOR_MAPPING.keys() if color in unique_colors]

    # Filter and sort data to only include unique colors and shapes
    filtered_stacked_data = stacked_data.loc[unique_shapes, sorted_colors]

    # Create Stacked Bar Chart
    plt.figure(figsize=(20, 15), facecolor="white")
    filtered_stacked_data.plot(
        kind="bar",
        stacked=True,
        figsize=(14, 10),
        color=[COLOR_MAPPING[color] for color in sorted_colors],
        edgecolor="black",
        linewidth=0.5,
    )

    # Aesthetics
    plt.title("Stacked Bar Chart of Images Based on Color and Shape", fontsize=16)
    plt.xlabel("Shape", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title="Color", fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.25, color="grey", axis="y")

    # If save_path is specified, save the plot, otherwise display it
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        # Show the plot
        plt.show()
