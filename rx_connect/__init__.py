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
CKPT_DIR = ROOT_DIR / "checkpoints"
"""The checkpoint directory.
"""
PIPELINES_DIR = PROJECT_DIR / "pipelines"
"""The pipelines directory.
"""
CACHE_DIR = ROOT_DIR / ".cache"
"""The cache directory where artifacts loaded from the remote server are stored.
"""
SERVER_IP = "172.23.72.41"
"""The remote server IP address. Default to the current AI Lab GPU server.
"""
