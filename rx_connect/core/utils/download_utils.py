import hashlib
import re
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import gdown
from tqdm import tqdm

from rx_connect.tools.logging import setup_logger

logger = setup_logger()

__all__: Sequence[str] = ("calculate_md5", "download_and_extract_archive", "download_url", "extract_archive")


def calculate_md5(fpath: Union[str, Path], chunk_size: int = 1024 * 1024) -> Optional[str]:
    """Calculate the MD5 hash of a file.

    The file is opened specified by `file_path` in binary mode and reads it in chunks, updating
    the MD5 hash object with each chunk. This approach allows for the efficient processing of
    large files without consuming excessive memory.

    Args:
        fpath (str or Path): Path to the file.
        chunk_size (int): Chunk size in bytes.

    Returns:
        The MD5 hash of the file, as a 32-character hex string. None if the file does not exist
        or an error occurred while reading the file.
    """
    fpath = Path(fpath)
    md5_hash = hashlib.md5(usedforsecurity=False)
    total_size = fpath.stat().st_size

    try:
        with open(fpath, "rb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=fpath.name) as pbar:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    md5_hash.update(chunk)
                    pbar.update(len(chunk))
        return md5_hash.hexdigest()
    except FileNotFoundError:
        logger.error(f"File not found: {fpath}")
        return None
    except Exception as e:
        logger.error(f"Error occurred while calculating MD5 hash of {fpath}: {e}")
        return None


def _get_google_drive_file_id(url: str) -> Optional[str]:
    """Extracts file ID from Google Drive URL.

    Args:
        url (str): Google Drive URL.

    Returns:
        The extracted file ID, or None if the URL format is unrecognized.
    """
    # Match the pattern of the file ID in the URL.
    match = re.search(r"https://drive\.google\.com/file/d/([^/]+)/", url)

    # Returning the matched file ID if found, or None if not found
    return match.group(1) if match else None


def download_url(
    url: str,
    root: Union[str, Path],
    filename: str,
    md5: Optional[str] = None,
    postprocess: Optional[Callable[[str, Optional[str]], List[str]]] = None,
) -> Path:
    """Download a file from a URL and place it in root directory. If the file is not in the cache or the
    MD5 hash doen't match, it will be downloaded from the URL specified and stored in the cache directory.

    Args:
        url (str): URL of the file to download.
        root (str, Path): Directory to place downloaded file in.
        filename (str): Name to save the file under.
        md5 (str, optional): MD5 checksum of the download. If None, the MD5 checksum will not be verified.
        postprocess (callable, optional): Function called with filename as postprocess. This can be used
            to extract the contents of the downloaded file. For example, if the downloaded file is a zip
            file, the contents of the zip file can be extracted by specifying `postprocess=gdown.extractall`.

    Returns:
        The path to the downloaded file.
    """
    if Path(filename).suffix == "":
        raise ValueError(f"Filename must have an extension: {filename}")

    # Convert root to Path object
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # check if file is located on Google Drive
    file_id = _get_google_drive_file_id(url)

    # If file_id is not None, construct the URL to download the file from Google Drive
    if file_id is not None:
        url = f"https://drive.google.com/uc?id={file_id}"

    # Download the file
    output_path = gdown.cached_download(url, str(root / filename), md5=md5, postprocess=postprocess)

    return Path(output_path)


def _extract_zip_member(
    zip_file: zipfile.ZipFile, member: str, destination_dir: Union[str, Path], pbar: tqdm
) -> None:
    """Extracts a member from a zip file and updates the progress bar.

    Args:
        zip_file (zipfile.ZipFile): The zip file to extract from.
        member (str): The name of the member to extract.
        destination_dir (Union[str, Path]): The directory to extract to.
        pbar (tqdm): The progress bar to update.
    """
    zip_file.extract(member, str(destination_dir))
    pbar.update(1)


def _extract_tar_member(
    tar_file: tarfile.TarFile, member_name: str, destination_dir: Union[str, Path], pbar: tqdm
) -> None:
    """Extracts a member from a tar file and updates the progress bar.

    Args:
        tar_file (tarfile.TarFile): The tar file to extract from.
        member_name (str): The name of the member to extract.
        destination_dir (Union[str, Path]): The directory to extract to.
        pbar (tqdm): The progress bar to update.
    """
    member = tar_file.getmember(member_name)
    tar_file.extract(member, str(destination_dir))
    pbar.update(1)


def extract_archive(
    archive_path: Union[str, Path], destination_dir: Union[str, Path], remove_finished: bool = False
) -> None:
    """Extracts an archive file to a specified directory. Supports .zip, .tar.gz, and .tar file formats.

    Args:
        archive_path (Union[str, Path]): The path to the archive file.
        destination_dir (Union[str, Path]): The directory to extract the archive to.
        remove_finished (bool): If True, the archive file will be deleted after extraction.

    Raises:
        ValueError: If the archive format is not supported.
    """
    archive_path, destination_dir = Path(archive_path), Path(destination_dir)

    def extract_files(
        file: Union[zipfile.ZipFile, tarfile.TarFile], file_list_func: Callable, extract_func: Callable
    ) -> None:
        # Calculate the total number of files in the file
        total_files = len(file_list_func(file))

        # Initialize a progress bar with the total number of files
        with tqdm(total=total_files, desc="Extracting files") as pbar:
            # Use ThreadPoolExecutor to extract files concurrently
            with ThreadPoolExecutor() as executor:
                # For each file in the file, submit a job to extract it
                for member in file_list_func(file):
                    executor.submit(extract_func, file, member, str(destination_dir), pbar)

    # Check the file extension of the archive_path
    if archive_path.suffix == ".zip":
        # If the file is a .zip file, open it with zipfile.ZipFile
        with zipfile.ZipFile(archive_path, "r") as zip_file:
            # Use the extract_files function to extract the .zip file
            extract_files(zip_file, zipfile.ZipFile.namelist, _extract_zip_member)

    elif archive_path.suffix == ".gz" and archive_path.stem.endswith(".tar"):
        # If the file is a .tar.gz file, open it with tarfile.open
        with tarfile.open(archive_path, "r:gz") as tar_file:
            # Use the extract_files function to extract the .tar.gz file
            extract_files(tar_file, tarfile.TarFile.getnames, _extract_tar_member)

    elif archive_path.suffix == ".tar":
        # If the file is a .tar file, open it with tarfile.open
        with tarfile.open(archive_path, "r:") as tar_file:
            # Use the extract_files function to extract the .tar file
            extract_files(tar_file, tarfile.TarFile.getnames, _extract_tar_member)

    else:
        raise ValueError("Unsupported archive format (only .zip, .tar.gz, and .tar are supported)")

    # Remove the archive file if remove_finished is True
    if remove_finished:
        archive_path.unlink()


def download_and_extract_archive(
    url: str,
    filename: str,
    download_root: Union[str, Path],
    extract_root: Optional[Union[str, Path]] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
) -> None:
    """Downloads and extracts a compressed file from a URL.

    Args:
        url (str): URL of the file to download.
        filename (str): Name to save the file under.
        download_root (str, Path): Directory to place downloaded file in.
        extract_root (str, Path, optional): Directory to extract the downloaded file to. If None, the
            downloaded file will be extracted to the same directory as the downloaded file.
        md5 (str, optional): MD5 checksum of the download. If None, the MD5 checksum will not be verified.
        remove_finished (bool): If True, the downloaded file will be deleted after extraction.
    """
    # Download the file
    fpath = download_url(url, download_root, filename, md5=md5)

    # If extract_root is None, set it to the same directory as the downloaded file
    extract_root = extract_root or download_root
    extract_root = Path(extract_root)

    # Extract the downloaded file
    logger.info(f"Extracting {filename} to {extract_root.absolute()}")
    extract_archive(fpath, extract_root, remove_finished)
