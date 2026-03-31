"""
FastAPI route definitions.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.agents.graph import run_echomaps_pipeline

log = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class LocationRequest(BaseModel):
    lat:           float = Field(..., ge=-90,   le=90)
    lon:           float = Field(..., ge=-180,  le=180)
    location_name: str   = Field(default="",   max_length=200)
    query:         str   = Field(default="Tell me the history of this place",
                                  max_length=500)


class Source(BaseModel):
    title:  str
    url:    str
    source: str
    score:  float


class Interpretation(BaseModel):
    persona:        str
    interpretation: str
    confidence:     float


class GraphNode(BaseModel):
    id:    str
    label: str
    type:  str


class GraphEdge(BaseModel):
    source:   str
    target:   str
    relation: str


class KnowledgeGraph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class LocationResponse(BaseModel):
    # Core narrative
    narrative:      str
    location_name:  str
    location_id:    str

    # Per-persona outputs
    interpretations: list[Interpretation]

    # RAG sources
    sources: list[Source]

    # Knowledge graph
    knowledge_graph: KnowledgeGraph


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/location", response_model=LocationResponse)
async def get_location_context(req: LocationRequest) -> LocationResponse:
    """Run the full EchoMaps LangGraph pipeline for a map coordinate.

    Returns a synthesised multi-perspective narrative, individual persona
    interpretations, RAG sources, and an extracted knowledge graph.
    """
    # Use provided name or fall back to coordinate string
    location_name = req.location_name.strip() or f"{req.lat:.4f}, {req.lon:.4f}"

    try:
        result: dict[str, Any] = run_echomaps_pipeline(req.lat, req.lon, location_name)
    except Exception as exc:
        log.exception("Pipeline error for (%.4f, %.4f): %s", req.lat, req.lon, exc)
        raise HTTPException(status_code=500, detail="Pipeline failed") from exc

    # ── Narrative ────────────────────────────────────────────────────────────
    narrative = result.get("final_narrative") or "No narrative could be generated."

    # ── Interpretations ──────────────────────────────────────────────────────
    interpretations = [
        Interpretation(
            persona=i.get("persona", ""),
            interpretation=i.get("interpretation", ""),
            confidence=float(i.get("confidence", 0.0)),
        )
        for i in result.get("interpretations", [])
    ]

    # ── Sources ──────────────────────────────────────────────────────────────
    sources = [
        Source(
            title=s.get("title", ""),
            url=s.get("url", ""),
            source=s.get("source", ""),
            score=float(s.get("score", 0.0)),
        )
        for s in result.get("sources", [])
    ]

    # ── Knowledge graph ──────────────────────────────────────────────────────
    raw_kg = result.get("knowledge_graph", {"nodes": [], "edges": []})
    knowledge_graph = KnowledgeGraph(
        nodes=[GraphNode(**n) for n in raw_kg.get("nodes", [])],
        edges=[GraphEdge(**e) for e in raw_kg.get("edges", [])],
    )

    return LocationResponse(
        narrative=narrative,
        location_name=location_name,
        location_id=result.get("location_id", ""),
        interpretations=interpretations,
        sources=sources,
        knowledge_graph=knowledge_graph,
    )
