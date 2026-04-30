"""Entry point for Render hosting (or any uvicorn runner).

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
    # or:
    python -m uvicorn main:app --host 0.0.0.0 --port $PORT

Also compatible with gunicorn:
    gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
"""
import sys
import os

# Add project root so src/ imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.inference.api import app

# Re-export for ASGI servers
__all__ = ["app"]
