import uvicorn
import argparse
import os
import sys

# 确保能导入同目录下的 main.py
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动BART摘要后端服务")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址")
    parser.add_argument("--port", type=int, default=8000, help="绑定端口")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")

    args = parser.parse_args()

    # 直接导入 app 对象，避免 uvicorn 的字符串导入失败
    from main import app 

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1,
        log_level="info"
    )