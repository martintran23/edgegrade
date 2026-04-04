"""
Card Grading AI: FastAPI entrypoint.

Run locally::

    uvicorn app.main:app --reload --app-dir backend
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import analyze
from app.core.config import settings

app = FastAPI(title=settings.api_title, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for deploys and local sanity checks."""
    return {"status": "ok"}
