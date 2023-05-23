# Tools
In this folder, there are multiple tool functions that'll be shared among several modules.
</br></br>

## Data fetch from remote server (Download)
Requirement - password-less SSH
1. Generate ssh keys (hit <enter> to the end)
```
ssh-keygen
```
2. Send ssh public key to remote server
```
ssh-copy-id <user_name>@10.231.51.79
```
</br>

Example 1 - files
```
from rx_connect.tools import data_tools

remote_file_name = "/home/wtsain4/test_remote_file.txt"
print(data_tools.fetch_from_remote(remote_file_name))
```
Function returns the local path "`local_cache/test_remote_file.txt`", where the file is cached.
</br>

Example 2 - files
```
from rx_connect.tools import data_tools

remote_folder_name = "/home/wtsain4/test_remote_folder"
print(data_tools.fetch_from_remote(remote_folder_name))
```
Function returns the local path "`local_cache/test_remote_folder`", where the folder and all the files it contains are cached recursively.
