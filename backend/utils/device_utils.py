import os
import torch
from backend.core.logging_config import get_logger
from backend.config import settings

logger = get_logger(__name__)

def get_device() -> str:
    """Detect available device and return 'cuda'/'mps'/'cpu'."""
    
    # 如果配置指定了设备且不是 auto，优先使用
    if settings.DEVICE != "auto":
        if settings.DEVICE == "cuda" and torch.cuda.is_available():
            logger.info(f"Using configured device: {settings.DEVICE}")
            return "cuda"
        elif settings.DEVICE == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info(f"Using configured device: {settings.DEVICE}")
            return "mps"
        elif settings.DEVICE == "cpu":
            logger.info(f"Using configured device: {settings.DEVICE}")
            return "cpu"
        else:
            logger.warning(f"Configured device '{settings.DEVICE}' not available, auto-detecting...")
    
    # 强制设备（用于调试）
    if settings.FORCE_DEVICE:
        force_dev = settings.FORCE_DEVICE.lower()
        if force_dev in ("cuda", "gpu"):
            if torch.cuda.is_available():
                logger.info("FORCE_DEVICE=cuda and torch.cuda is available, using GPU")
                return "cuda"
            else:
                logger.warning("FORCE_DEVICE=cuda but torch.cuda is not available, checking other devices")
    
    # 自动检测
    if torch.cuda.is_available():
        logger.info("Auto-detected device: cuda")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Auto-detected device: mps")
        return "mps"
    else:
        logger.info("Auto-detected device: cpu")
        return "cpu"