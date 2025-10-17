import asyncio
import time
from fastapi import APIRouter, HTTPException, Request
from backend.models import (
    SummaryRequest,
    SummaryResponse,
    BatchSummaryRequest,
    BatchSummaryResponse,
)

router = APIRouter()

@router.post("/summarize", response_model=SummaryResponse)
async def summarize_text(request: Request, body: SummaryRequest):
    svc = request.app.state.summarization_service

    if not svc.model_loaded:
        raise HTTPException(status_code=503, detail="The model is not fully loaded, please try again later.")

    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")

    start_time = time.time()

    try:
        summary = await asyncio.get_event_loop().run_in_executor(
            svc.thread_pool,
            svc.generate_summary,
            body.text,
            body.max_length,
            body.min_length,
            body.num_beams,
            body.length_penalty,
        )

        processing_time = time.time() - start_time
        original_length = len(body.text.split())
        summary_length = len(summary.split())
        compression_ratio = original_length / summary_length if summary_length > 0 else 0

        return SummaryResponse(
            summary=summary,
            processing_time=processing_time,
            model_used=svc.model_name,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=round(compression_ratio, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

@router.post("/summarize/batch", response_model=BatchSummaryResponse)
async def batch_summarize(request: Request, body: BatchSummaryRequest):
    svc = request.app.state.summarization_service

    if not svc.model_loaded:
        raise HTTPException(status_code=503, detail="The model is not fully loaded, please try again later.")

    if not body.texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty.")

    start_time = time.time()

    try:
        summaries = []
        for text in body.texts:
            if text.strip():
                summary = await asyncio.get_event_loop().run_in_executor(
                    svc.thread_pool,
                    svc.generate_summary,
                    text,
                    body.max_length,
                    body.min_length,
                    # 保持与原逻辑一致：批量接口未传递 num_beams/length_penalty
                )
                summaries.append(summary)
            else:
                summaries.append("")

        processing_time = time.time() - start_time

        return BatchSummaryResponse(
            summaries=summaries,
            processing_time=processing_time,
            total_texts=len(body.texts),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch summary generation failed: {str(e)}")