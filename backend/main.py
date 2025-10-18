from fastapi import FastAPI
from backend.config import settings
from backend.core.middleware import setup_cors
from backend.core.logger import get_logger
from backend.services.summarization_service import SummarizationService

from backend.api.routes.root import router as root_router
from backend.api.routes.health import router as health_router
from backend.api.routes.summarization import router as summarization_router

logger = get_logger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_TITLE,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
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
        logger.info(f"Starting {settings.APP_TITLE} v{settings.APP_VERSION}...")
        logger.info(f"Configuration: {settings.display_settings()}")
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
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL,
    )