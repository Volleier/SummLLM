# backend/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration,
    BartTokenizer
)
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 数据模型
class SummaryRequest(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 30
    num_beams: int = 4
    length_penalty: float = 2.0


class SummaryResponse(BaseModel):
    summary: str
    processing_time: float
    model_used: str
    original_length: int
    summary_length: int
    compression_ratio: float
    status: str = "success"


class BatchSummaryRequest(BaseModel):
    texts: List[str]
    max_length: int = 150
    min_length: int = 30


class BatchSummaryResponse(BaseModel):
    summaries: List[str]
    processing_time: float
    total_texts: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_name: str


# BART-large-CNN 后端服务
class BartSummarizationBackend:
    def __init__(self):
        self.app = FastAPI(
            title="BART-large-CNN 文本摘要API",
            description="基于BART-large-CNN模型的文本摘要服务",
            version="1.0.0"
        )

        # 模型配置
        # 优先从环境变量 LOCAL_MODEL_PATH 加载本地模型（例如 E:\Project\SummLLM\backend\models\bart.large.cnn）
        default_local = os.path.join(os.path.dirname(__file__), "models", "bart.large.cnn")
        self.model_name = os.environ.get("LOCAL_MODEL_PATH", default_local)
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self.model_loaded = False

        # 线程池用于处理并发请求
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        self.setup_middleware()
        self.setup_routes()

        # 启动时预加载模型
        self.app.add_event_handler("startup", self.startup_event)
        self.app.add_event_handler("shutdown", self.shutdown_event)

    def _get_device(self):
        """检测可用设备"""
        # 支持通过环境变量强制设备（例如 FORCE_DEVICE=cuda）
        force_dev = os.environ.get("FORCE_DEVICE", "").lower()
        if force_dev in ("cuda", "gpu"):
            if torch.cuda.is_available():
                logger.info("环境变量 FORCE_DEVICE=cuda，且 torch.cuda 可用，使用 GPU")
                return "cuda"
            else:
                logger.warning("环境变量 FORCE_DEVICE=cuda，但 torch.cuda 不可用，继续检测其它设备")

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def setup_middleware(self):
        """设置中间件"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # 生产环境中应限制来源
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_routes(self):
        """设置API路由"""

        @self.app.get("/", include_in_schema=False)
        async def root():
            return {"message": "BART-large-CNN 文本摘要服务运行中"}

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            return HealthResponse(
                status="healthy",
                model_loaded=self.model_loaded,
                device=self.device,
                model_name=self.model_name
            )

        @self.app.post("/summarize", response_model=SummaryResponse)
        async def summarize_text(request: SummaryRequest):
            """单文本摘要接口"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="模型未加载完成，请稍后重试")

            if not request.text.strip():
                raise HTTPException(status_code=400, detail="文本内容不能为空")

            start_time = time.time()

            try:
                # 在线程池中运行模型推理
                summary = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    self._generate_summary,
                    request.text,
                    request.max_length,
                    request.min_length,
                    request.num_beams,
                    request.length_penalty
                )

                processing_time = time.time() - start_time
                original_length = len(request.text.split())
                summary_length = len(summary.split())
                compression_ratio = original_length / summary_length if summary_length > 0 else 0

                return SummaryResponse(
                    summary=summary,
                    processing_time=processing_time,
                    model_used=self.model_name,
                    original_length=original_length,
                    summary_length=summary_length,
                    compression_ratio=round(compression_ratio, 2)
                )

            except Exception as e:
                logger.error(f"摘要生成失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"摘要生成失败: {str(e)}")

        @self.app.post("/summarize/batch", response_model=BatchSummaryResponse)
        async def batch_summarize(request: BatchSummaryRequest):
            """批量文本摘要接口"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="模型未加载完成，请稍后重试")

            if not request.texts:
                raise HTTPException(status_code=400, detail="文本列表不能为空")

            start_time = time.time()

            try:
                # 批量处理
                summaries = []
                for text in request.texts:
                    if text.strip():
                        summary = await asyncio.get_event_loop().run_in_executor(
                            self.thread_pool,
                            self._generate_summary,
                            text,
                            request.max_length,
                            request.min_length
                        )
                        summaries.append(summary)
                    else:
                        summaries.append("")

                processing_time = time.time() - start_time

                return BatchSummaryResponse(
                    summaries=summaries,
                    processing_time=processing_time,
                    total_texts=len(request.texts)
                )

            except Exception as e:
                logger.error(f"批量摘要生成失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"批量摘要生成失败: {str(e)}")

    async def startup_event(self):
        """应用启动时加载模型"""
        logger.info("正在启动BART-large-CNN摘要服务...")
        await self.load_model()

    async def shutdown_event(self):
        """应用关闭时清理资源"""
        logger.info("正在关闭服务...")
        self.thread_pool.shutdown(wait=True)

    async def load_model(self):
        """异步加载模型"""
        try:
            logger.info(f"开始加载模型: {self.model_name}")
            logger.info(f"使用设备: {self.device}")

            # 方法1: 使用pipeline (推荐，更简单)
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self.model_loaded = True
                logger.info("BART-large-CNN模型加载成功!")
                return
            except Exception as e:
                logger.exception(f"尝试在 {self.device} 加载模型失败: {e}")

            # 如果原先期望使用 GPU，但失败则回退到 CPU 再尝试一次（方便本地调试）
            if self.device == "cuda":
                logger.info("GPU 加载失败，尝试回退到 CPU 加载模型...")
                try:
                    self.device = "cpu"
                    self.summarizer = pipeline(
                        "summarization",
                        model=self.model_name,
                        tokenizer=self.model_name,
                        device=-1,
                        torch_dtype=torch.float32
                    )
                    self.model_loaded = True
                    logger.info("已在 CPU 上成功加载模型（回退加载）。")
                    return
                except Exception as e2:
                    logger.exception(f"回退到 CPU 加载模型也失败: {e2}")

            self.model_loaded = False
            logger.error("模型加载最终失败，请检查模型目录（是否为 Hugging Face 格式），或确认 PyTorch CUDA 可用。")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.model_loaded = False
            # 这里可以添加重试逻辑

    def _generate_summary(self, text: str, max_length: int = 150, min_length: int = 30,
                          num_beams: int = 4, length_penalty: float = 2.0) -> str:
        """生成摘要的核心方法"""
        try:
            # 文本预处理
            cleaned_text = self._preprocess_text(text)

            # 使用pipeline生成摘要
            if hasattr(self, 'summarizer'):
                result = self.summarizer(
                    cleaned_text,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    do_sample=False  # 使用beam search，不采样
                )
                return result[0]['summary_text']

            # 或者使用模型直接生成
            elif self.model and self.tokenizer:
                inputs = self.tokenizer(
                    cleaned_text,
                    max_length=1024,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs['input_ids'],
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=num_beams,
                        length_penalty=length_penalty,
                        early_stopping=True
                    )

                summary = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                return summary

            else:
                raise Exception("模型未正确加载")

        except Exception as e:
            logger.error(f"摘要生成错误: {e}")
            raise e

    def _preprocess_text(self, text: str, max_input_length: int = 1024) -> str:
        """文本预处理"""
        import re

        # 清理文本
        text = re.sub(r'\s+', ' ', text)  # 合并多余空白字符
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)  # 移除特殊字符
        text = text.strip()

        # 限制输入长度
        if len(text) > max_input_length:
            text = text[:max_input_length] + "..."

        return text


# 创建应用实例
app_backend = BartSummarizationBackend()
app = app_backend.app

if __name__ == "__main__":
    import uvicorn

    # 启动服务
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发时启用热重载
        workers=1,  # 由于模型较大，建议使用1个worker
        log_level="info"
    )