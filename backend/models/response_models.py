from pydantic import BaseModel
from typing import List

class SummaryResponse(BaseModel):
    summary: str
    processing_time: float
    model_used: str
    original_length: int
    summary_length: int
    compression_ratio: float
    status: str = "success"

class BatchSummaryResponse(BaseModel):
    summaries: List[str]
    processing_time: float
    total_texts: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_name: str