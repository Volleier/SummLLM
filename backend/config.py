import os
from pathlib import Path
from typing import Literal

class Settings:
    """Application configuration settings."""
    
    # ==================== Application Settings ====================
    APP_TITLE: str = "BART-large-CNN Text Summarization API"
    APP_DESCRIPTION: str = "Text summarization service based on BART-large-CNN model"
    APP_VERSION: str = "1.0.0"
    
    # ==================== Server Settings ====================
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    RELOAD: bool = os.getenv("RELOAD", "true").lower() == "true"
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    
    # ==================== CORS Settings ====================
    CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]
    
    # ==================== Model Settings ====================
    # 在线 HuggingFace repo ID
    HF_REPO_ID: str = os.getenv("HF_REPO_ID", "facebook/bart-large-cnn")
    
    # 本地模型路径（优先级最高）
    LOCAL_MODEL_PATH: str | None = os.getenv("LOCAL_MODEL_PATH", "E:/models/bart-large-cnn")
    
    # 设备选择：auto（自动检测）/ cuda / cpu / mps
    DEVICE: Literal["auto", "cuda", "cpu", "mps"] = os.getenv("DEVICE", "auto")  # type: ignore
    
    # 强制使用指定设备（用于调试）
    FORCE_DEVICE: str | None = os.getenv("FORCE_DEVICE")
    
    # ==================== Model Generation Settings ====================
    # 默认生成参数
    DEFAULT_MAX_LENGTH: int = int(os.getenv("DEFAULT_MAX_LENGTH", "150"))
    DEFAULT_MIN_LENGTH: int = int(os.getenv("DEFAULT_MIN_LENGTH", "30"))
    DEFAULT_NUM_BEAMS: int = int(os.getenv("DEFAULT_NUM_BEAMS", "4"))
    DEFAULT_LENGTH_PENALTY: float = float(os.getenv("DEFAULT_LENGTH_PENALTY", "2.0"))
    
    # 输入文本最大长度
    MAX_INPUT_LENGTH: int = int(os.getenv("MAX_INPUT_LENGTH", "1024"))
    
    # ==================== Service Settings ====================
    # 线程池最大工作线程数
    THREAD_POOL_MAX_WORKERS: int = int(os.getenv("THREAD_POOL_MAX_WORKERS", "4"))
    
    # ==================== Cache Settings ====================
    # HuggingFace 缓存目录
    HF_HOME: str | None = os.getenv("HF_HOME")
    HF_HUB_CACHE: str | None = os.getenv("HF_HUB_CACHE")
    TRANSFORMERS_CACHE: str | None = os.getenv("TRANSFORMERS_CACHE")
    
    # ==================== Helper Methods ====================
    @classmethod
    def get_local_model_path(cls) -> Path | None:
        """获取并验证本地模型路径."""
        if not cls.LOCAL_MODEL_PATH:
            return None
        path = Path(cls.LOCAL_MODEL_PATH)
        if path.exists() and path.is_dir():
            return path
        return None
    
    @classmethod
    def get_cache_roots(cls) -> list[Path]:
        """获取所有可能的 HuggingFace 缓存根目录."""
        roots = []
        
        # 从环境变量获取
        for cache_path in [cls.HF_HOME, cls.HF_HUB_CACHE, cls.TRANSFORMERS_CACHE]:
            if cache_path:
                p = Path(cache_path)
                roots.append(p)
                # HF_HOME 通常包含 hub 子目录
                hub = p / "hub"
                if hub.exists():
                    roots.append(hub)
        
        # 默认缓存位置
        default_home = Path.home() / ".cache" / "huggingface" / "hub"
        roots.extend([default_home, default_home.parent])
        
        return roots
    
    @classmethod
    def display_settings(cls) -> dict:
        """返回配置摘要（用于日志输出）."""
        return {
            "app_version": cls.APP_VERSION,
            "host": cls.HOST,
            "port": cls.PORT,
            "device": cls.DEVICE,
            "hf_repo_id": cls.HF_REPO_ID,
            "local_model_path": cls.LOCAL_MODEL_PATH,
            "max_input_length": cls.MAX_INPUT_LENGTH,
            "thread_pool_workers": cls.THREAD_POOL_MAX_WORKERS,
        }


# 全局配置实例
settings = Settings()