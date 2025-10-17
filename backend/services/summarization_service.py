import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from concurrent.futures import ThreadPoolExecutor

from backend.config import settings
from backend.core.logger import get_logger
from backend.utils.device_utils import get_device
from backend.utils.text_utils import preprocess_text

logger = get_logger(__name__)

def _to_posix(path_str: str) -> str:
    return Path(path_str).resolve().as_posix()

class SummarizationService:
    """BART-large-CNN backend service wrapper - 仅从配置的本地路径加载模型，路径无效则抛错。"""

    def __init__(self) -> None:
        self.repo_id = settings.HF_REPO_ID
        # 这里直接调用 config 中的验证方法；若路径无效会抛出 FileNotFoundError，启动失败
        self.local_model_dir: Path = settings.get_local_model_path()
        self.device = get_device()
        self.model_loaded: bool = False
        self.model_name: str = str(self.local_model_dir)

        self.model = None
        self.tokenizer = None
        self.summarizer = None
        self.thread_pool = ThreadPoolExecutor(max_workers=settings.THREAD_POOL_MAX_WORKERS)

    async def load_model(self) -> None:
        """只从 LOCAL_MODEL_PATH 加载模型；若加载失败抛出异常以便上层处理（启动失败）。"""
        model_path = _to_posix(str(self.local_model_dir))
        logger.info(f"Loading model from configured LOCAL_MODEL_PATH: {model_path} using device {self.device}")

        try:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            # 显式加载 tokenizer 与 model（local_files_only=True）
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype)
            # 使用已加载的 model/tokenizer 创建 pipeline（避免 pipeline 内部处理文件路径）
            self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer,
                                      device=0 if self.device == "cuda" else -1)
            self.model_loaded = True
            self.model_name = f"{self.repo_id} (local: {model_path})"
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.exception(f"Failed to load model from LOCAL_MODEL_PATH: {model_path} - {e}")
            # 抛出以让应用启动失败并上报错误（符合“路径没有模型则提交错误”的要求）
            raise RuntimeError(f"Failed to load model from LOCAL_MODEL_PATH: {model_path}") from e

    def shutdown(self) -> None:
        self.thread_pool.shutdown(wait=True)

    def generate_summary(
        self,
        text: str,
        max_length: int | None = None,
        min_length: int | None = None,
        num_beams: int | None = None,
        length_penalty: float | None = None,
    ) -> str:
        # 使用配置的默认值
        if max_length is None:
            max_length = settings.DEFAULT_MAX_LENGTH
        if min_length is None:
            min_length = settings.DEFAULT_MIN_LENGTH
        if num_beams is None:
            num_beams = settings.DEFAULT_NUM_BEAMS
        if length_penalty is None:
            length_penalty = settings.DEFAULT_LENGTH_PENALTY

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

            else:
                raise RuntimeError("Model is not loaded correctly")
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            raise