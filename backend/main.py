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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Data models: Pydantic definitions for requests and responses (used for input validation and automatic docs)
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


# BART-large-CNN backend service wrapper class
class BartSummarizationBackend:
    def __init__(self):
        # Initialize FastAPI app and basic attributes
        self.app = FastAPI(
            title="BART-large-CNN Text Summarization API",
            description="Text summarization service based on BART-large-CNN model",
            version="1.0.0"
        )

        # Model path: prefer environment variable, otherwise use project default path
        default_local = os.path.join(os.path.dirname(__file__), "models", "bart.large.cnn")
        self.model_name = os.environ.get("LOCAL_MODEL_PATH", default_local)
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()  # Auto-detect device (cuda/mps/cpu)
        self.model_loaded = False

        # Thread pool to run inference tasks in threads to avoid blocking the event loop
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        self.setup_middleware()
        self.setup_routes()

        # Attach startup and shutdown event handlers
        self.app.add_event_handler("startup", self.startup_event)
        self.app.add_event_handler("shutdown", self.shutdown_event)

    def _get_device(self):
        """Detect available device and return 'cuda'/'mps'/'cpu'"""
        # Support forcing device via environment variable (e.g., FORCE_DEVICE=cuda)
        force_dev = os.environ.get("FORCE_DEVICE", "").lower()
        if force_dev in ("cuda", "gpu"):
            if torch.cuda.is_available():
                logger.info("Environment variable FORCE_DEVICE=cuda and torch.cuda is available, using GPU")
                return "cuda"
            else:
                logger.warning("Environment variable FORCE_DEVICE=cuda but torch.cuda is not available, checking other devices")

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def setup_middleware(self):
        """Set up CORS middleware (allow all origins for easier development)"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Should restrict origins in production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_routes(self):
        """Define API routes and handlers"""

        @self.app.get("/", include_in_schema=False)
        async def root():
            # Root path for a simple health check
            return {"message": "BART-large-CNN Text Summarization API is running"}

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            # Return service and model load status
            return HealthResponse(
                status="healthy",
                model_loaded=self.model_loaded,
                device=self.device,
                model_name=self.model_name
            )

        @self.app.post("/summarize", response_model=SummaryResponse)
        async def summarize_text(request: SummaryRequest):
            """Single-text summarization endpoint"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="The model is not fully loaded, please try again later.")

            if not request.text.strip():
                raise HTTPException(status_code=400, detail="Text content cannot be empty.")

            start_time = time.time()

            try:
                # Run model inference in thread pool to avoid blocking event loop
                summary = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    self._generate_summary,
                    request.text,
                    request.max_length,
                    request.min_length,
                    request.num_beams,
                    request.length_penalty
                )

                # Compute processing time and compression ratio statistics
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
                logger.error(f"Summary generation failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

        @self.app.post("/summarize/batch", response_model=BatchSummaryResponse)
        async def batch_summarize(request: BatchSummaryRequest):
            """Batch summarization endpoint"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="The model is not fully loaded, please try again later.")

            if not request.texts:
                raise HTTPException(status_code=400, detail="Text list cannot be empty.")

            start_time = time.time()

            try:
                # Submit each text to thread pool (could be changed to concurrent batch processing)
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
                logger.error(f"Batch summary generation failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Batch summary generation failed: {str(e)}")

    async def startup_event(self):
        """Load model at application startup"""
        logger.info("Starting BART-large-CNN text summarization service...")
        await self.load_model()

    async def shutdown_event(self):
        """Cleanup resources (like thread pool) on application shutdown"""
        logger.info("Shutting down the service...")
        self.thread_pool.shutdown(wait=True)

    async def load_model(self):
        """Asynchronously load the model, try pipeline first and log errors on failure"""
        try:
            logger.info(f"Starting to load the model: {self.model_name}")
            logger.info(f"Using device: {self.device}")

            # Method 1: use pipeline (recommended, wraps tokenizer + model)
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self.model_loaded = True
                logger.info("BART-large-CNNModel loaded successfully!")
                return
            except Exception as e:
                # Log failure of loading attempt; may try fallback to CPU below
                logger.exception(f"Failed to load model on {self.device}: {e}")

            # If GPU load fails, try fallback to CPU (useful for local debugging)
            if self.device == "cuda":
                logger.info("GPU loading failed, trying to fallback to CPU...")
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
                    logger.info("Successfully loaded model on CPU (fallback).")
                    return
                except Exception as e2:
                    logger.exception(f"Fallback to CPU loading also failed: {e2}")

            # If all load methods fail, mark as not loaded and log error
            self.model_loaded = False
            logger.error("Model loading failed.")

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            self.model_loaded = False
            # Add retry logic here if desired

    def _generate_summary(self, text: str, max_length: int = 150, min_length: int = 30,
                          num_beams: int = 4, length_penalty: float = 2.0) -> str:
        """Core method to generate summary (can run in a thread)"""
        try:
            # Preprocess text
            cleaned_text = self._preprocess_text(text)

            # Prefer using pipeline interface to generate summary
            if hasattr(self, 'summarizer'):
                result = self.summarizer(
                    cleaned_text,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    do_sample=False  # Use beam search (no sampling)
                )
                return result[0]['summary_text']

            # If no pipeline, try manual generation using model + tokenizer
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
                # Raise exception when no model is loaded
                raise Exception("The model didn't load correctly")

        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            raise e

    def _preprocess_text(self, text: str, max_input_length: int = 1024) -> str:
        """Simple text cleaning and length truncation"""
        import re

        # Collapse extra whitespace and remove most special characters
        text = re.sub(r'\s+', ' ', text)  # Collapse multiple whitespace characters
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)  # Remove special characters
        text = text.strip()

        # Limit input max length, truncate and append ellipsis if too long
        if len(text) > max_input_length:
            text = text[:max_input_length] + "..."

        return text


# Create application instance and export app (for uvicorn or other ASGI servers)
app_backend = BartSummarizationBackend()
app = app_backend.app

if __name__ == "__main__":
    import uvicorn

    # Local development startup command (only runs when script executed directly)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable hot reload during development
        workers=1,  # Use 1 worker due to large model
        log_level="info"
    )