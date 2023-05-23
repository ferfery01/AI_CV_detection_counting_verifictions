import subprocess
from pathlib import Path


def fetch_from_remote(
    remote_path: str,
    server_ip: str = "10.231.51.79",
    local_cashe_folder: str = "local_cache",
) -> str:
    """
    Fetch data from remote server if not yet cached.

    Note: requires password-less SSH. Steps:
        1. $ ssh-keygen
            (you can hit <enter> to the end)
        2. $ ssh-copy-id <user_name>@10.231.51.79


    Args:
        remote_path (str): Full path or relative path (to home) of the remote file.
        server_ip (str): The remote server IP address. Default to the AI Lab GPU server at 10.231.51.79.
        local_cashe_folder (str): The desired name of the caching folder.

    Returns:
        str: Local file name after caching.
    """
    Path(local_cashe_folder).mkdir(exist_ok=True)

    local_path = Path(local_cashe_folder + "/" + Path(remote_path).name)
    if not local_path.exists():
        subprocess.run(["scp", "-r", f"{server_ip}:{remote_path}", local_path])
    assert local_path.exists()

    return str(local_path)
