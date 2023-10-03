import concurrent.futures
from collections import Counter
from pathlib import Path
from typing import List

import click
from tqdm import tqdm, trange

from rx_connect import ROOT_DIR
from rx_connect.core.images.io import download_image
from rx_connect.core.utils.io_utils import get_matching_files_in_dir
from rx_connect.core.utils.str_utils import str_to_hash
from rx_connect.dataset.scrape_nih_images import extract_image_urls
from rx_connect.dataset.utils import (
    LAYOUT_METADATA,
    Layouts,
    load_consumer_image_df_by_layout,
)
from rx_connect.generator.metadata_filter import parse_file_hash
from rx_connect.tools.logging import setup_logger
from rx_connect.tools.serialization import read_pickle, write_pickle

logger = setup_logger()


"""This script is designed to download original images corresponding to pills that are either
missing or poorly segmented after manual inspection. The script operates in the following steps:

1. Load a DataFrame containing metadata about consumer-grade images, filtered by the specified image layout.
2. Identify missing or poorly segmented images based on their file hash.
3. Download these identified images to a specified directory.

Behavior:
    - Utilizes multi-threading for efficient image downloading.
    - Provides a progress bar for real-time monitoring.
    - Cleans up images that are found but are no longer missing in the new dataset.

Example:
    python download_specific_image.py --input-dir ./existing_images --download-dir ./new_images --layout LayoutType
"""


@click.command()
@click.option(
    "-i",
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the directory containing the downloaded images.",
)
@click.option(
    "-d",
    "--download-dir",
    default=ROOT_DIR / "data/Pill_Images",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the directory to save the downloaded images.",
)
@click.option(
    "-l",
    "--layout",
    required=True,
    type=click.Choice(Layouts.members()),
    help="""Select the Image layout to download. More information about the layout can be
    obtained at https://data.lhncbc.nlm.nih.gov/public/Pills/RxImageImageLayouts.docx""",
)
def main(input_dir: Path, download_dir: Path, layout: str) -> None:
    # Load the consumer grade images csv file and filter it based on the layout and missing hash
    metadata_df = load_consumer_image_df_by_layout(download_dir, layout=Layouts[layout])
    metadata_df["File_Hash"] = metadata_df.FileName.apply(lambda x: str_to_hash(x))
    metadata_df = metadata_df.set_index("File_Hash")

    # Get the list of all images and masks in the input directory
    images_path = get_matching_files_in_dir(input_dir / "images", "*.[jJ][pP][gG]")
    masks_path = get_matching_files_in_dir(input_dir / "masks", "*.[pP][nN][gG]")

    # Filter the metadata dataframe to only include the images that are missing
    images_hash = [parse_file_hash(p).split("_")[0] for p in images_path]
    masks_hash = [parse_file_hash(p).split("_")[0] for p in masks_path]
    missing_hash = list(set(masks_hash) - set(images_hash))
    missing_images = metadata_df.loc[missing_hash].FileName.tolist()

    # If the number of images in the directory is less than the max number of pills, then add the file
    # to the missing list
    file_count = Counter(images_hash)
    missing_images += [
        metadata_df.loc[file_hash].FileName
        for file_hash, count in file_count.items()
        if count != LAYOUT_METADATA[Layouts[layout]].max_pills
    ]
    logger.info(f"Number of missing images: {len(missing_images)}")

    # Get the list of missing image URLs
    image_url_path = download_dir / f"{layout}_image_urls_dict.pkl"
    if image_url_path.exists():
        file_name_image_url = read_pickle(download_dir / f"{layout}_image_urls_dict.pkl")
    else:
        image_urls: List[str] = []
        for idx in trange(1, 111, desc="Extracting image URLs"):
            project_url = (
                f"https://data.lhncbc.nlm.nih.gov/public/Pills/PillProjectDisc{idx}/images/index.html"
            )
            image_urls += extract_image_urls(project_url)
        file_name_image_url = {url.rsplit("/")[-1]: url for url in image_urls}
        write_pickle(file_name_image_url, image_url_path)

    # Get the list of image URLs to download
    urls_to_download = [file_name_image_url[img_name] for img_name in missing_images]
    assert len(urls_to_download) == len(
        missing_images
    ), "Number of URLs to download must match the number of missing files."

    # Create project folder if it doesn't exist
    image_dir = download_dir / "images" / layout
    image_dir.mkdir(parents=True, exist_ok=True)

    # Create a list of tuples containing the image URL and the path to save the image to
    # Ignore images that have already been downloaded
    img_info = []
    for url in urls_to_download:
        image_name = url.rsplit("/", 1)[-1]
        image_path = image_dir / image_name
        if not image_path.exists():
            img_info.append((url, image_path))

    # Delete the images that are not missing
    for img_path in image_dir.iterdir():
        if img_path.name not in missing_images:
            img_path.unlink()

    logger.info(f"Number of images to download: {len(img_info)}")
    # Use a ThreadPoolExecutor to download the images concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(lambda p: download_image(*p), info) for info in img_info]

        # Display the progress bar while the images are being downloaded
        for _ in tqdm(
            concurrent.futures.as_completed(futures), desc="Downloading images", total=len(futures)
        ):
            # results from future are ignored in this case
            pass


if __name__ == "__main__":
    main()
