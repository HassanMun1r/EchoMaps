"""
FastAPI route definitions.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.rag.retriever import retrieve_context_for_location

log = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class LocationRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    query: str = Field(default="Tell me the history of this place", max_length=500)


class Source(BaseModel):
    title: str
    url: str
    source: str
    score: float


class LocationResponse(BaseModel):
    context: str
    sources: list[Source]
    location_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/location", response_model=LocationResponse)
async def get_location_context(req: LocationRequest) -> LocationResponse:
    """Retrieve historical/cultural context chunks for a map coordinate."""
    try:
        result = retrieve_context_for_location(req.lat, req.lon, req.query)
    except Exception as exc:
        log.exception("retriever error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve context") from exc

    chunks: list[str] = result.get("chunks", [])
    if not chunks:
        context_text = "No historical information found for this location."
    else:
        context_text = "\n\n---\n\n".join(chunks)

    sources = [
        Source(
            title=s.get("title", ""),
            url=s.get("url", ""),
            source=s.get("source", ""),
            score=s.get("score", 0.0),
        )
        for s in result.get("sources", [])
    ]

    return LocationResponse(
        context=context_text,
        sources=sources,
        location_id=result.get("location_id", ""),
    )
