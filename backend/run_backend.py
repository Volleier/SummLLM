import uvicorn
import argparse
import os
import sys

# Ensure main.py in the same directory can be imported
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the BART summarization backend service")
    parser.add_argument("--host", default="0.0.0.0", help="bind address")
    parser.add_argument("--port", type=int, default=8000, help="bind port")
    parser.add_argument("--reload", action="store_true", help="enable hot reload for development")

    args = parser.parse_args()

    # Import the app object directly to avoid uvicorn's string import failure
    from main import app 

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1,
        log_level="info"
    )