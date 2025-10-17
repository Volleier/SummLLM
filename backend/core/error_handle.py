import sys
import traceback
import datetime
from typing import Optional

from backend.core.logger import get_logger

_err_logger = get_logger("error_handler")

def log_exception(exc: BaseException, context: Optional[str] = None) -> None:
    """
    记录异常到日志文件，并将异常信息与堆栈打印到控制台（stderr）。
    - exc: 捕获到的异常对象
    - context: 可选的上下文说明
    """
    ts = datetime.datetime.now().isoformat()
    ctx = f"[{context}] " if context else ""
    # 记录到本地日志（包含堆栈）
    _err_logger.error(f"{ts} - {ctx}Exception: {exc}", exc_info=True)
    # 同时把可读的错误与堆栈输出到控制台（stderr）
    print(f"{ts} - {ctx}Exception: {exc}", file=sys.stderr)
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)

def log_message(message: str) -> None:
    """记录普通错误/警告消息到日志，并打印到控制台（stderr）。"""
    ts = datetime.datetime.now().isoformat()
    _err_logger.error(f"{ts} - {message}")
    print(f"{ts} - {message}", file=sys.stderr)