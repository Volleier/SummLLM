import os
from pathlib import Path
from typing import List, Literal

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
    # 在线 HuggingFace repo ID（保留）
    HF_REPO_ID: str = os.getenv("HF_REPO_ID", "facebook/bart-large-cnn")
    
    # 本地模型路径（必须配置或使用默认），项目要求：只有指定路径且路径内有模型才通过
    LOCAL_MODEL_PATH: str = os.getenv("LOCAL_MODEL_PATH", "E:/models/bart-large-cnn")
    
    # 设备选择：auto（自动检测）/ cuda / cpu / mps
    DEVICE: Literal["auto", "cuda", "cpu", "mps"] = os.getenv("DEVICE", "auto")  # type: ignore
    
    # 强制使用指定设备（用于调试）
    FORCE_DEVICE: str | None = os.getenv("FORCE_DEVICE")
    
    # ==================== Model Generation Settings ====================
    DEFAULT_MAX_LENGTH: int = int(os.getenv("DEFAULT_MAX_LENGTH", "150"))
    DEFAULT_MIN_LENGTH: int = int(os.getenv("DEFAULT_MIN_LENGTH", "30"))
    DEFAULT_NUM_BEAMS: int = int(os.getenv("DEFAULT_NUM_BEAMS", "4"))
    DEFAULT_LENGTH_PENALTY: float = float(os.getenv("DEFAULT_LENGTH_PENALTY", "2.0"))
    
    # 输入文本最大长度
    MAX_INPUT_LENGTH: int = int(os.getenv("MAX_INPUT_LENGTH", "1024"))
    
    # ==================== Service / Logging Settings ====================
    THREAD_POOL_MAX_WORKERS: int = int(os.getenv("THREAD_POOL_MAX_WORKERS", "4"))
    BACKEND_ROOT: str = os.getenv("BACKEND_ROOT", str(Path(__file__).resolve().parents[0]))
    LOG_DIR_NAME: str = os.getenv("LOG_DIR_NAME", "logs")

    # ==================== Helper Methods (简化) ====================
    @classmethod
    def get_local_model_path(cls) -> Path:
        """
        返回必须存在且包含模型文件的本地模型目录 Path。
        - 如果路径不存在或目录内不包含模型文件（config.json + (model.safetensors | pytorch_model.bin)）
          则抛出 FileNotFoundError。
        """
        path = Path(cls.LOCAL_MODEL_PATH)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(
                f"LOCAL_MODEL_PATH 指定的路径不存在或不是目录: {path}. "
                f"请通过环境变量 LOCAL_MODEL_PATH 指定正确的本地模型目录。"
            )
        has_config = (path / "config.json").exists()
        has_model = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
        if not (has_config and has_model):
            raise FileNotFoundError(
                f"LOCAL_MODEL_PATH 目录中缺少模型文件（需要 config.json 与 model.safetensors 或 pytorch_model.bin）: {path}"
            )
        return path

    @classmethod
    def get_backend_root(cls) -> Path:
        return Path(cls.BACKEND_ROOT)

    @classmethod
    def get_log_dir(cls) -> Path:
        p = cls.get_backend_root() / cls.LOG_DIR_NAME
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    @classmethod
    def display_settings(cls) -> dict:
        return {
            "app_version": cls.APP_VERSION,
            "host": cls.HOST,
            "port": cls.PORT,
            "device": cls.DEVICE,
            "hf_repo_id": cls.HF_REPO_ID,
            "local_model_path": cls.LOCAL_MODEL_PATH,
            "max_input_length": cls.MAX_INPUT_LENGTH,
            "thread_pool_workers": cls.THREAD_POOL_MAX_WORKERS,
            "backend_root": str(cls.get_backend_root()),
            "log_dir": str(cls.get_log_dir()),
        }

    def get_cache_roots(self) -> List[Path]:
        """
        返回一个按优先级排列的 cache 根目录列表，供 _find_local_snapshot 搜索 snapshots。
        优先使用环境变量（HF_HOME、TRANSFORMERS_CACHE），然后是常见默认位置。
        """
        roots: List[Path] = []

        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            roots.append(Path(hf_home))

        tf_cache = os.environ.get("TRANSFORMERS_CACHE")
        if tf_cache:
            roots.append(Path(tf_cache))

        # 常见 huggingface 缓存位置（hub / models）
        roots.append(Path.home() / ".cache" / "huggingface" / "hub")
        roots.append(Path.home() / ".cache" / "huggingface" / "models")

        # 去重并返回绝对路径
        seen = set()
        final: List[Path] = []
        for p in roots:
            try:
                p = p.resolve()
            except Exception:
                p = Path(p)
            if p not in seen:
                seen.add(p)
                final.append(p)
        return final


# 全局配置实例
settings = Settings()