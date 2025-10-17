from fastapi import FastAPI
from backend.core.middleware import setup_cors
from backend.core.logging_config import get_logger
from backend.services.summarization_service import SummarizationService

from backend.api.routes.root import router as root_router
from backend.api.routes.health import router as health_router
from backend.api.routes.summarization import router as summarization_router

logger = get_logger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(
        title="BART-large-CNN Text Summarization API",
        description="Text summarization service based on BART-large-CNN model",
        version="1.0.0",
    )

    # Service instance
    svc = SummarizationService()
    app.state.summarization_service = svc

    # Middleware
    setup_cors(app)

    # Routers
    app.include_router(root_router)
    app.include_router(health_router)
    app.include_router(summarization_router)

    # Lifecycle
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting BART-large-CNN text summarization service...")
        await svc.load_model()

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down the service...")
        svc.shutdown()

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
        log_level="info",
    )