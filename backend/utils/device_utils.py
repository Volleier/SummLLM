import os
import torch
from backend.core.logging_config import get_logger

logger = get_logger(__name__)

def get_device() -> str:
    """Detect available device and return 'cuda'/'mps'/'cpu'."""
    force_dev = os.environ.get("FORCE_DEVICE", "").lower()
    if force_dev in ("cuda", "gpu"):
        if torch.cuda.is_available():
            logger.info("Environment variable FORCE_DEVICE=cuda and torch.cuda is available, using GPU")
            return "cuda"
        else:
            logger.warning("Environment variable FORCE_DEVICE=cuda but torch.cuda is not available, checking other devices")

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"