from pydantic import BaseModel
from typing import List

class SummaryRequest(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 30
    num_beams: int = 4
    length_penalty: float = 2.0

class BatchSummaryRequest(BaseModel):
    texts: List[str]
    max_length: int = 150
    min_length: int = 30