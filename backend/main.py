"""
EchoMaps FastAPI application entry point.

Run with:
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="EchoMaps API",
    description="Geospatial multi-agent RAG — historical and cultural narratives for any location.",
    version="0.1.0",
)

# Allow the frontend (opened as a local file or dev server) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)
