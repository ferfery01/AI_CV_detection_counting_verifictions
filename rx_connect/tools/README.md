# Tools
In this folder, there are multiple tool functions that'll be shared among several modules.

## Data fetch from remote server (Download)
**Requirements:** For the functions in this file to work, you need to be one a VPN and have password-less SSH set up. Steps:
1. Generate ssh keys (hit <enter> to the end)

    ```
    ssh-keygen
    ```
2. Send ssh public key to remote server

    ```
    ssh-copy-id <user_name>@172.23.72.41
    ```

Example 1 - files
```
from rx_connect.tools import data_tools

remote_file_name = "/home/wtsain4/test_remote_file.txt"
print(data_tools.fetch_from_remote(remote_file_name))
```
Function returns the local path "`.cache/test_remote_file.txt`", where the file is cached.

Example 2 - folders
```
from rx_connect.tools import data_tools

remote_folder_name = "/home/wtsain4/test_remote_folder"
print(data_tools.fetch_from_remote(remote_folder_name))
```
Function returns the local path "`.cache/test_remote_folder`", where the folder and all the files it contains are cached recursively.
