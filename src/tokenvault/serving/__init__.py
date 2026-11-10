"""HTTP serving. Requires the ``serve`` extra (fastapi, uvicorn)."""

from tokenvault.serving.api import AsyncEngine, create_app

__all__ = ["AsyncEngine", "create_app"]
