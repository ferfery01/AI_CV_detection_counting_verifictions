from pathlib import Path

SHARED_REMOTE_DIR = "/media/RxConnectShared"
"""The shared remote directory name.
"""
SHARED_EPILL_DATA_DIR = "/media/RxConnectShared/ePillID/pills/"
"""The shared ePillID data directory.
"""
PROJECT_DIR = Path(__file__).parent
"""The project directory.
"""
ROOT_DIR = PROJECT_DIR.parent
"""The root directory.
"""
CACHE_DIR = ROOT_DIR / ".cache/"
"""The cache directory where artifacts loaded from the remote server are stored.
"""
SERVER_IP = "10.231.51.79"
"""The remote server IP address. Default to the current AI Lab GPU server.
"""
