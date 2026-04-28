"""Start the FastAPI anomaly detection server.

Usage:
    python scripts/serve_api.py
    python scripts/serve_api.py --host 0.0.0.0 --port 8000
    python scripts/serve_api.py --reload          # dev mode

API docs available at http://localhost:8000/docs after startup.
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)

    print(f"Starting API server at http://{args.host}:{args.port}")
    print(f"Interactive docs:  http://{args.host}:{args.port}/docs")
    print(f"Demo UI:           http://{args.host}:{args.port}/demo")

    uvicorn.run(
        "src.inference.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
