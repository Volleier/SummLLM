from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
import time

from backend.core.logger import get_logger
from backend.core import error_handle

logger = get_logger(__name__)

def setup_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )

    # 注册请求日志中间件：记录开始/结束与耗时，发生异常时交由 error_handle 处理
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        logger.info("请求开始: method=%s url=%s client=%s", request.method, str(request.url), request.client.host if request.client else None)
        try:
            response = await call_next(request)
        except Exception as exc:
            error_handle.log_exception(exc, context=f"request {request.method} {request.url}")
            raise
        duration = time.time() - start
        logger.info("请求结束: method=%s url=%s status=%s duration=%.3fs", request.method, str(request.url), getattr(response, "status_code", None), duration)
        return response