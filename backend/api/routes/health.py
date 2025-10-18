from fastapi import APIRouter, Request
from backend.models import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    svc = request.app.state.summarization_service
    return HealthResponse(
        status="healthy",
        model_loaded=svc.model_loaded,
        device=svc.device,
        model_name=svc.model_name,
    )