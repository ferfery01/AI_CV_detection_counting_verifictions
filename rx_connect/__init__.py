from pathlib import Path

SHARED_REMOTE_DIR = Path("/media/RxConnectShared")
"""The shared remote directory name.
"""
SHARED_REMOTE_CKPT_DIR = SHARED_REMOTE_DIR / "checkpoints"
"""The shared remote checkpoint directory.
"""
SHARED_EPILL_DATA_DIR = SHARED_REMOTE_DIR / "ePillID" / "pills"
"""The shared ePillID data directory.
"""
SHARED_RXIMAGE_DATA_DIR = SHARED_REMOTE_DIR / "RxImage"
"""The shared RxImage data directory.
"""
SHARED_RXIMAGEV2_DATA_DIR = SHARED_REMOTE_DIR / "RxImageV2"
"""The shared RxImageV2 data directory.
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
