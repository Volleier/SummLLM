import os
import time
from pathlib import Path
import torch
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor

from backend.core.logging_config import get_logger
from backend.utils.device_utils import get_device
from backend.utils.text_utils import preprocess_text

logger = get_logger(__name__)

def _to_posix(path_str: str) -> str:
    return Path(path_str).resolve().as_posix()

def _is_model_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    has_config = (p / "config.json").exists()
    has_model = (p / "pytorch_model.bin").exists() or (p / "model.safetensors").exists()
    return has_config and has_model

def _find_local_snapshot(repo_id: str) -> Path | None:
    # 在常见缓存目录中查找 snapshots 目录
    repo_folder = f"models--{repo_id.replace('/', '--')}"
    candidates = []

    env_roots = [os.getenv("TRANSFORMERS_CACHE"), os.getenv("HF_HUB_CACHE"), os.getenv("HF_HOME")]
    roots: list[Path] = []
    for r in env_roots:
        if r:
            roots.append(Path(r))
            # HF_HOME 一般包含 hub 子目录
            hub = Path(r) / "hub"
            if hub.exists():
                roots.append(hub)

    # 兜底：用户未设置环境变量时，也尝试默认缓存位置
    default_home = Path.home() / ".cache" / "huggingface" / "hub"
    roots.extend([default_home, default_home.parent])  # 同时尝试 hub 与其父目录

    for root in roots:
        snapshots_dir = root / repo_folder / "snapshots"
        if snapshots_dir.exists() and snapshots_dir.is_dir():
            for snap in snapshots_dir.iterdir():
                if snap.is_dir():
                    # 仅接受包含模型文件的 snapshot
                    if _is_model_dir(snap):
                        candidates.append(snap)

    if not candidates:
        return None

    # 选最近修改的 snapshot
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

class SummarizationService:
    """BART-large-CNN backend service wrapper."""

    def __init__(self) -> None:
        # 在线 repo id（默认）
        self.repo_id = os.environ.get("HF_REPO_ID", "facebook/bart-large-cnn")

        # 优先从环境变量指定的本地目录加载
        env_local = os.environ.get("LOCAL_MODEL_PATH", "").strip()
        self.local_model_dir: Path | None = Path(env_local) if env_local else None

        self.device = get_device()
        self.model_loaded: bool = False

        self.model = None
        self.tokenizer = None

        self.summarizer = None
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    async def load_model(self) -> None:
        """Load model: prefer local dir; else try local cache snapshot; else online repo."""
        try:
            logger.info(f"Using device: {self.device}")

            local_dir: Path | None = None

            if self.local_model_dir and _is_model_dir(self.local_model_dir):
                local_dir = self.local_model_dir
                logger.info(f"Loading from LOCAL_MODEL_PATH: {local_dir}")
            else:
                if self.local_model_dir:
                    logger.warning(f"LOCAL_MODEL_PATH not a valid model dir: {self.local_model_dir}")
                # 尝试从缓存找到可用的 snapshot
                snap = _find_local_snapshot(self.repo_id)
                if snap:
                    local_dir = snap
                    logger.info(f"Loading from local snapshot: {snap}")

            # 按顺序尝试：本地 -> 在线；GPU -> CPU
            def _build_pipeline(model_arg, use_local: bool, device_str: str):
                return pipeline(
                    "summarization",
                    model=model_arg,
                    tokenizer=model_arg,
                    device=0 if device_str == "cuda" else -1,
                    torch_dtype=torch.float16 if device_str == "cuda" else torch.float32,
                    local_files_only=use_local,
                )

            # 1) 本地目录优先（如有）
            if local_dir:
                model_path = _to_posix(str(local_dir))
                try:
                    self.summarizer = _build_pipeline(model_path, use_local=True, device_str=self.device)
                    self.model_loaded = True
                    logger.info(f"Model loaded locally from: {model_path}")
                    return
                except Exception as e:
                    logger.exception(f"Local model load failed at {model_path}: {e}")

            # 2) 在线加载（GPU）
            try:
                self.summarizer = _build_pipeline(self.repo_id, use_local=False, device_str=self.device)
                self.model_loaded = True
                logger.info(f"Model loaded from hub: {self.repo_id}")
                return
            except Exception as e:
                logger.exception(f"Failed to load from hub on {self.device}: {e}")

            # 3) 在线加载（CPU 回退）
            if self.device == "cuda":
                try:
                    self.summarizer = _build_pipeline(self.repo_id, use_local=False, device_str="cpu")
                    self.model_loaded = True
                    self.device = "cpu"
                    logger.info(f"Successfully loaded from hub on CPU (fallback).")
                    return
                except Exception as e2:
                    logger.exception(f"Fallback to CPU loading also failed: {e2}")

            self.model_loaded = False
            logger.error("Model loading failed.")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            self.model_loaded = False

    def shutdown(self) -> None:
        self.thread_pool.shutdown(wait=True)

    def generate_summary(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        num_beams: int = 4,
        length_penalty: float = 2.0,
    ) -> str:
        try:
            cleaned_text = preprocess_text(text)

            if self.summarizer is not None:
                result = self.summarizer(
                    cleaned_text,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    do_sample=False,
                )
                return result[0]["summary_text"]

            elif self.model is not None and self.tokenizer is not None:
                inputs = self.tokenizer(
                    cleaned_text,
                    max_length=1024,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=num_beams,
                        length_penalty=length_penalty,
                        early_stopping=True,
                    )

                summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return summary

            else:
                raise Exception("The model didn't load correctly")

        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            raise