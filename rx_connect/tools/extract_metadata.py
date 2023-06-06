from typing import List, Optional, TypedDict

import requests
from bs4 import BeautifulSoup

from rx_connect.tools.logging import setup_logger

logger = setup_logger()


_TAGS: List[str] = ["Imprint", "Color", "Shape", "Strength"]


class DrugMetadata(TypedDict, total=False):
    Name: Optional[str]
    Imprint: Optional[str]
    Color: Optional[str]
    Shape: Optional[str]
    Strength: Optional[str]


def parse_drug_info(ndc_code: str) -> Optional[BeautifulSoup]:
    """Fetches drug information from drugs.com based on the NDC code.

    Args:
        ndc_code (str): The NDC (National Drug Code) of the drug.

    Returns:
        Optional[BeautifulSoup]: Parsed HTML content as a BeautifulSoup object,
            or None if there was an error accessing the website.
    """
    soup: Optional[BeautifulSoup] = None
    try:
        url = f"https://www.drugs.com/imprints.php?ndc={ndc_code}"
        response = requests.get(url)
        response.raise_for_status()  # Raises exception for non-200 status codes

        soup = BeautifulSoup(response.content, "html.parser")
    except requests.RequestException as e:
        logger.error("Error accessing the website:", e)

    return soup


def parse_drug_name(soup: BeautifulSoup) -> Optional[str]:
    """Extracts the drug name from the provided BeautifulSoup object.

    Args:
        soup (BeautifulSoup): Parsed HTML content as a BeautifulSoup object.

    Returns:
        Optional[str]: The drug name if found, or None if not found.
    """
    a_tag = soup.find("a", {"class": "ddc-text-size-small"})
    drug_name = a_tag.text if a_tag is not None else None

    return drug_name


def parse_tags(soup: BeautifulSoup, string: str) -> Optional[str]:
    """Extracts unique tag values from the provided BeautifulSoup object based on
    the given string.

    Args:
        soup (BeautifulSoup): Parsed HTML content as a BeautifulSoup object.
        string (str): The string used to filter the tags.

    Returns:
        A string containing unique tag values joined with "; ", or None if no tags
        were found.
    """
    s_tags = soup.find_all("dl", {"class": "ddc-text-size-small"})
    values: List[str] = []
    for tag in s_tags:
        tag = tag.find("dt", string=string)
        if tag:
            value = tag.find_next_sibling("dd").text
            values.append(value)

    return "; ".join(set(values)) if len(values) > 0 else None


def parse_drug_metadata(ndc_code: str) -> DrugMetadata:
    """Parses drug metadata based on the NDC code.

    Args:
        ndc_code (str): The NDC (National Drug Code) of the drug.

    Returns:
        A dictionary containing the parsed drug metadata. The keys are the metadata
        tags, and the values are the corresponding values.
    """
    metadata: DrugMetadata = {}

    soup = parse_drug_info(ndc_code)
    if soup is not None:
        metadata = {tag: parse_tags(soup, tag) for tag in _TAGS}  # type: ignore
        metadata["Name"] = parse_drug_name(soup)

    return metadata
