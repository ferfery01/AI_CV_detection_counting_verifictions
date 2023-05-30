import subprocess
from pathlib import Path
from typing import Union


def fetch_from_remote(
    remote_path: str,
    server_ip: str = "10.231.51.79",
    local_cache_folder: str = "local_cache",
    ignore_exist: bool = False,
) -> str:
    """
    Fetch data from remote server if not yet cached.

    Note: requires password-less SSH. Steps:
        1. $ ssh-keygen
            (you can hit <enter> to the end)
        2. $ ssh-copy-id <user_name>@10.231.51.79


    Args:
        remote_path (str): Full path or relative path (to home) of the remote file/folder.
        server_ip (str): The remote server IP address. Default to the AI Lab GPU server at 10.231.51.79.
        local_cache_folder (str): The desired name of the caching folder.
        ignore_exist (bool): Enforce rsync checking time stamp if file/folder already exists.

    Returns:
        str: Local file name after caching.
    """
    Path(local_cache_folder).mkdir(exist_ok=True)

    local_path = Path(local_cache_folder + "/" + Path(remote_path).name)
    if not local_path.exists() or ignore_exist:
        subprocess.run(["rsync", "-ruq", f"{server_ip}:{remote_path}", local_path])
    assert local_path.exists(), f"Error: Failed to cache {remote_path} from {server_ip}."

    return str(local_path)


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
