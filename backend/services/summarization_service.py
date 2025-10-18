import time
from pathlib import Path
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from concurrent.futures import ThreadPoolExecutor

from backend.config import settings
from backend.core.logger import get_logger
from backend.core import error_handle
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
    """在常见缓存目录中查找 snapshots 目录."""
    repo_folder = f"models--{repo_id.replace('/', '--')}"
    candidates = []

    roots = settings.get_cache_roots()

    for root in roots:
        snapshots_dir = root / repo_folder / "snapshots"
        if snapshots_dir.exists() and snapshots_dir.is_dir():
            for snap in snapshots_dir.iterdir():
                if snap.is_dir() and _is_model_dir(snap):
                    candidates.append(snap)

    if not candidates:
        return None

    # 选最近修改的 snapshot
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

class SummarizationService:
    """BART-large-CNN backend service wrapper."""

    def __init__(self) -> None:
        self.repo_id = settings.HF_REPO_ID
        self.local_model_dir = settings.get_local_model_path()
        self.device = get_device()
        self.model_loaded: bool = False
        self.model_name: str = settings.HF_REPO_ID  # 添加 model_name 属性

        self.model = None
        self.tokenizer = None
        self.summarizer = None
        
        self.thread_pool = ThreadPoolExecutor(max_workers=settings.THREAD_POOL_MAX_WORKERS)

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
                snap = _find_local_snapshot(self.repo_id)
                if snap:
                    local_dir = snap
                    logger.info(f"Loading from local snapshot: {snap}")

            # 修改点：显式加载 tokenizer/model，避免 local_files_only 被误传为 model_kwargs
            def _build_pipeline(model_arg, use_local: bool, device_str: str):
                dtype = torch.float16 if device_str == "cuda" else torch.float32
                # 1) 明确用 from_pretrained 加载 tokenizer 与 model（支持 local_files_only）
                tokenizer = AutoTokenizer.from_pretrained(model_arg, local_files_only=use_local, use_fast=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_arg, local_files_only=use_local, torch_dtype=dtype)
                # 2) 将已加载的 model/tokenizer 传给 pipeline（避免 pipeline 内部传递 local_files_only）
                return pipeline(
                    "summarization",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if device_str == "cuda" else -1,
                )

            # 1) 本地目录优先
            if local_dir:
                model_path = _to_posix(str(local_dir))
                try:
                    self.summarizer = _build_pipeline(model_path, use_local=True, device_str=self.device)
                    self.model_loaded = True
                    self.model_name = f"{self.repo_id} (local: {model_path})"  # 更新 model_name
                    logger.info(f"Model loaded locally from: {model_path}")
                    return
                except Exception as e:
                    error_handle.log_exception(e, context=f"Local model load failed at {model_path}")

            # 2) 在线加载（GPU/指定设备）
            try:
                self.summarizer = _build_pipeline(self.repo_id, use_local=False, device_str=self.device)
                self.model_loaded = True
                self.model_name = f"{self.repo_id} (online)"  # 更新 model_name
                logger.info(f"Model loaded from hub: {self.repo_id}")
                return
            except Exception as e:
                error_handle.log_exception(e, context=f"Failed to load from hub on {self.device}")

            # 3) 在线加载（CPU 回退）
            if self.device == "cuda":
                try:
                    self.summarizer = _build_pipeline(self.repo_id, use_local=False, device_str="cpu")
                    self.model_loaded = True
                    self.device = "cpu"
                    self.model_name = f"{self.repo_id} (online, CPU fallback)"  # 更新 model_name
                    logger.info("Successfully loaded from hub on CPU (fallback).")
                    return
                except Exception as e2:
                    error_handle.log_exception(e2, context="Fallback to CPU loading also failed")

            self.model_loaded = False
            self.model_name = f"{self.repo_id} (failed to load)"  # 加载失败时的 model_name
            error_handle.log_message("Model loading failed.")
        except Exception as e:
            error_handle.log_exception(e, context="Model loading top-level error")
            self.model_loaded = False
            self.model_name = f"{self.repo_id} (error)"

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

            elif self.model is not None and self.tokenizer is not None:
                inputs = self.tokenizer(
                    cleaned_text,
                    max_length=settings.MAX_INPUT_LENGTH,
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
            error_handle.log_exception(e, context="generate_summary")
            raise