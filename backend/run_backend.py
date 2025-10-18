import uvicorn
from backend.config import settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("backend start: env=%s", "development")  # 可替换为真实环境/配置
    try:
        uvicorn.run(
            "backend.main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.RELOAD,
            workers=settings.WORKERS,
            log_level=settings.LOG_LEVEL,
        )
    except Exception as e:
        logger.exception("后端启动失败: %s", e)
        raise
    finally:
        logger.info("后端停止")