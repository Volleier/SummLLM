import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import datetime

from backend.config import settings

# 日志目录：从 config 获取后端根目录并在其下创建 logs 文件夹
LOG_DIR = settings.get_log_dir()
# 使用时间戳命名日志文件，格式：backend_YYYYmmdd_HHMMSS.log
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"backend_{timestamp}.log"

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for h in list(root_logger.handlers):
    root_logger.removeHandler(h)

# Rotating file handler
file_handler = RotatingFileHandler(
    filename=str(LOG_FILE),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
root_logger.addHandler(file_handler)

# 防止其他 logging 配置将日志再次路由到控制台
root_logger.propagate = False

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)