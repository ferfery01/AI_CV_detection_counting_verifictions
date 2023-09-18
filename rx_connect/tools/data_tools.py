import subprocess
from pathlib import Path
from typing import List, Union

from rx_connect import CACHE_DIR, SERVER_IP
from rx_connect.tools import is_remote_dir
from rx_connect.tools.logging import setup_logger

logger = setup_logger()

"""NOTE: For the functions in this file to work, you need to be one a VPN and have
password-less SSH set up. Steps:
    1. $ ssh-keygen (you can hit <enter> to the end)
    2. $ ssh-copy-id <user_name>@172.23.72.41
"""


def fetch_from_remote(
    remote_path: Union[str, Path],
    *,
    server_ip: str = SERVER_IP,
    cache_dir: Union[str, Path] = CACHE_DIR,
    ignore_exist: bool = False,
    timeout: int = 30,
) -> Path:
    """Fetch a file/folder from a remote server if this function is executed outside the remote server
    (e.g. on a local machine), otherwise it returns the original path. The file/folder will be cached
    in the cache_dir. If the file/folder already exists locally, then it will not be fetched
    unless `ignore_exist` is set to True.

    Args:
        remote_path (str, Path): Full path or relative path (to home) of the remote file/folder.
        server_ip (str): The remote server IP address. Default to the AI Lab GPU server at 172.23.72.41.
        cache_dir (str, Path): Path to the folder where cached files are stored.
        ignore_exist (bool): Enforce rsync checking time stamp if file/folder already exists.
        timeout (int): Timeout in seconds for the rsync command.

    Returns:
        Path: Path to the cached file/folder. If on the remote server, then it returns the original path.
    """
    remote_path = Path(remote_path)

    # Return the original path, if on the remote server
    if not is_remote_dir(remote_path):
        return remote_path

    # Create the cache directory, if it does not exist
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)

    # Fetch the file/folder from the remote server, if not in cache or if ignoring existence
    local_path = cache_dir / remote_path.name
    if not local_path.exists() or ignore_exist:
        try:
            result = subprocess.run(
                ["rsync", "-azq", f"{server_ip}:{remote_path}", local_path], timeout=timeout
            )
        except subprocess.TimeoutExpired as e:
            logger.error(f"Timeout expired when fetching {remote_path} from {server_ip}. Are you on VPN?")
            raise e

        if result.returncode == 23:
            raise FileNotFoundError(f"{remote_path} not found on {server_ip}.")
        elif result.returncode != 0:
            raise ValueError(f"Failed to fetch {remote_path} from {server_ip}.")

    return local_path


def fetch_file_paths_from_remote_dir(
    remote_dir: Union[str, Path], *, server_ip: str = SERVER_IP, timeout: int = 30
) -> List[Path]:
    """Fetch the list of files path in a remote directory.

    Args:
        remote_dir (str, Path): Full path or relative path (to home) of the remote directory.
        server_ip (str): The remote server IP address. Default to the AI Lab GPU server at
        timeout (int): Timeout in seconds for the ssh command.

    Returns:
        List[Path]: List of file paths in the remote directory.
    """
    remote_dir = Path(remote_dir)

    # Get the names of all of the files in the remote directory. It ignores any subdirectories.
    try:
        result = subprocess.run(
            ["ssh", server_ip, f"ls -p {remote_dir} | grep -v /"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        logger.error(
            f"Operation timed out when fetching file paths from {remote_dir} from {server_ip}. Are you on VPN?"
        )
        raise e

    if result.returncode != 0:
        raise ValueError(
            f"Failed to get the list of files in {remote_dir} from {server_ip}.\n"
            f"Error message: '{result.stderr.strip()}'"
        )

    # Get the list of all files in the remote directory
    remote_files = result.stdout.strip().split("\n")
    logger.info(f"Found {len(remote_files)} files in {remote_dir} from {server_ip}.")

    return [remote_dir / file_name for file_name in remote_files]


def push_to_remote(
    local_path: Union[str, Path],
    remote_storage_folder: str,
    *,
    server_ip: str = SERVER_IP,
) -> str:
    """
    Push local data to remote server.

    Note: requires password-less SSH. Steps:
        1. $ ssh-keygen
            (you can hit <enter> to the end)
        2. $ ssh-copy-id <user_name>@172.23.72.41

    Args:
        local_path (str or Path): Full path or relative path (to current) of the local file/folder.
        remote_storage_folder (str): The remote parent folder to put the file/folder in.
        server_ip (str): The remote server IP address. Default to the AI Lab GPU server at 172.23.72.41.

    Returns:
        str: Remote file/folder name after uploading.
    """
    local_path_obj = Path(local_path) if isinstance(local_path, str) else local_path
    assert local_path_obj.exists(), f"Error: {local_path_obj} not found."
    remote_folder_exist = (
        subprocess.run(
            ["ssh", server_ip, f"test -e {remote_storage_folder} && test -d {remote_storage_folder}"],
            capture_output=True,
        ).returncode
        == 0
    )
    assert remote_folder_exist, f"Error: Remote folder {remote_storage_folder} doesn't exist on {server_ip}."

    remote_path = f"{remote_storage_folder}/{local_path_obj.name}"
    subprocess.run(["rsync", "-rq", local_path_obj, f"{server_ip}:{remote_path}"])

    return remote_path
