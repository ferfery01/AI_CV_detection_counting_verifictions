import subprocess
from pathlib import Path
from typing import List, Union

from rx_connect.tools.logging import setup_logger

logger = setup_logger()

"""NOTE: For the functions in this file to work, you need to be one a VPN and have
password-less SSH set up. Steps:
    1. $ ssh-keygen (you can hit <enter> to the end)
    2. $ ssh-copy-id <user_name>@10.231.51.79
"""


def fetch_from_remote(
    remote_path: Union[str, Path],
    server_ip: str = "10.231.51.79",
    cache_dir: Union[str, Path] = "local_cache",
    ignore_exist: bool = False,
    timeout: int = 30,
) -> str:
    """
    Fetch data from remote server if not yet cached.

    Note: requires password-less SSH. Steps:
        1. $ ssh-keygen
            (you can hit <enter> to the end)
        2. $ ssh-copy-id <user_name>@10.231.51.79

    Args:
        remote_path (str, Path): Full path or relative path (to home) of the remote file/folder.
        server_ip (str): The remote server IP address. Default to the AI Lab GPU server at 10.231.51.79.
        cache_dir (str, Path): Path to the folder where cached files are stored.
        ignore_exist (bool): Enforce rsync checking time stamp if file/folder already exists.
        timeout (int): Timeout in seconds for the rsync command.

    Returns:
        str: Local file name after caching.
    """
    remote_path = Path(remote_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)

    local_path = cache_dir / remote_path.name
    if not local_path.exists() or ignore_exist:
        try:
            subprocess.run(["rsync", "-ruq", f"{server_ip}:{remote_path}", local_path], timeout=timeout)
        except subprocess.TimeoutExpired as e:
            logger.error(f"Timeout expired when fetching {remote_path} from {server_ip}. Are you on VPN?")
            raise e
    assert local_path.exists(), f"Error: Failed to cache {remote_path} from {server_ip}."

    return str(local_path)


def fetch_file_paths_from_remote_folder(
    remote_dir: Union[str, Path], server_ip: str = "10.231.51.79", timeout: int = 30
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
    server_ip: str = "10.231.51.79",
) -> str:
    """
    Push local data to remote server.

    Note: requires password-less SSH. Steps:
        1. $ ssh-keygen
            (you can hit <enter> to the end)
        2. $ ssh-copy-id <user_name>@10.231.51.79

    Args:
        local_path (str or Path): Full path or relative path (to current) of the local file/folder.
        remote_storage_folder (str): The remote parent folder to put the file/folder in.
        server_ip (str): The remote server IP address. Default to the AI Lab GPU server at 10.231.51.79.

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
