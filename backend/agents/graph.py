"""
LangGraph pipeline for EchoMaps.

State flow:
  fetch_context → run_historians → synthesize → extract_entities → END

Exposed entry point:
  run_echomaps_pipeline(lat, lon, location_name) -> dict
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypedDict

from groq import Groq
from langgraph.graph import END, START, StateGraph

from backend.agents.historian import run_persona_agent
from backend.agents.personas import PERSONAS
from backend.agents.synthesizer import synthesis_agent
from backend.config import GROQ_API_KEY
from backend.knowledge.extractor import extract_entities
from backend.knowledge.graph_builder import build_knowledge_graph
from backend.rag.retriever import retrieve_context_for_location

log = logging.getLogger(__name__)


# ── State schema ──────────────────────────────────────────────────────────────

class EchoMapsState(TypedDict, total=False):
    # Inputs
    lat:             float
    lon:             float
    location_name:   str

    # Set by fetch_context_node
    context_chunks:  list[str]
    sources:         list[dict[str, Any]]
    location_id:     str

    # Set by run_historians_node
    interpretations: list[dict[str, Any]]

    # Set by synthesize_node
    final_narrative: str

    # Set by extract_entities_node (populated in next commit)
    knowledge_graph: dict[str, Any]


# ── Groq client (module-level singleton) ─────────────────────────────────────

def _make_groq_client() -> Groq | None:
    if not GROQ_API_KEY:
        log.warning("GROQ_API_KEY not set — agent nodes will be skipped")
        return None
    return Groq(api_key=GROQ_API_KEY)


# ── Node implementations ──────────────────────────────────────────────────────

def fetch_context_node(state: EchoMapsState) -> EchoMapsState:
    """Node 1 — retrieve RAG context for the location."""
    lat  = state["lat"]
    lon  = state["lon"]
    log.info("[fetch_context] lat=%.4f lon=%.4f", lat, lon)

    result = retrieve_context_for_location(lat, lon, "Tell me the history of this place")

    return {
        **state,
        "context_chunks": result.get("chunks", []),
        "sources":        result.get("sources", []),
        "location_id":    result.get("location_id", ""),
    }


def run_historians_node(state: EchoMapsState) -> EchoMapsState:
    """Node 2 — run all 4 persona agents in parallel threads."""
    chunks        = state.get("context_chunks", [])
    location_name = state.get("location_name", "Unknown Location")

    groq_client = _make_groq_client()
    if groq_client is None or not chunks:
        log.warning("[run_historians] skipping — no Groq client or no chunks")
        return {
            **state,
            "interpretations": [
                {"persona": p, "interpretation": "Unavailable.", "confidence": 0.0}
                for p in PERSONAS
            ],
        }

    log.info("[run_historians] running %d personas in parallel", len(PERSONAS))

    results: dict[str, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(
                run_persona_agent,
                persona_name,
                chunks,
                location_name,
                groq_client,
            ): persona_name
            for persona_name in PERSONAS
        }
        for future in as_completed(futures):
            persona_name = futures[future]
            try:
                results[persona_name] = future.result()
            except Exception as exc:
                log.error("[run_historians] %s failed: %s", persona_name, exc)
                results[persona_name] = {
                    "persona": persona_name,
                    "interpretation": f"Agent error: {exc}",
                    "confidence": 0.0,
                }

    # Preserve definition order
    ordered = [results[p] for p in PERSONAS if p in results]

    return {**state, "interpretations": ordered}


def synthesize_node(state: EchoMapsState) -> EchoMapsState:
    """Node 3 — synthesise the four interpretations into one narrative."""
    interpretations = state.get("interpretations", [])
    location_name   = state.get("location_name", "Unknown Location")

    groq_client = _make_groq_client()
    if groq_client is None:
        return {**state, "final_narrative": "Synthesis unavailable (no Groq client)."}

    log.info("[synthesize] weaving %d interpretations", len(interpretations))
    narrative = synthesis_agent(interpretations, location_name, groq_client)

    return {**state, "final_narrative": narrative}


def extract_entities_node(state: EchoMapsState) -> EchoMapsState:
    """Node 4 — NER on the final narrative, then build the knowledge graph."""
    narrative     = state.get("final_narrative", "")
    location_id   = state.get("location_id", "unknown")
    location_name = state.get("location_name", "Unknown Location")

    # Also run NER over the raw context chunks for richer entity coverage
    context_text = " ".join(state.get("context_chunks", []))
    combined_text = f"{narrative}\n\n{context_text}".strip()

    log.info("[extract_entities] running NER on %d chars", len(combined_text))
    entities = extract_entities(combined_text)
    log.info("[extract_entities] found %d unique entities", len(entities))

    kg = build_knowledge_graph(entities, location_id, location_name)
    return {**state, "knowledge_graph": kg}


# ── Graph assembly ────────────────────────────────────────────────────────────

def _build_graph() -> Any:
    builder = StateGraph(EchoMapsState)

    builder.add_node("fetch_context",   fetch_context_node)
    builder.add_node("run_historians",  run_historians_node)
    builder.add_node("synthesize",      synthesize_node)
    builder.add_node("extract_entities", extract_entities_node)

    builder.add_edge(START,              "fetch_context")
    builder.add_edge("fetch_context",    "run_historians")
    builder.add_edge("run_historians",   "synthesize")
    builder.add_edge("synthesize",       "extract_entities")
    builder.add_edge("extract_entities", END)

    return builder.compile()


_GRAPH = _build_graph()


# ── Public entry point ────────────────────────────────────────────────────────

def run_echomaps_pipeline(
    lat: float,
    lon: float,
    location_name: str,
) -> dict[str, Any]:
    """Run the full EchoMaps agent pipeline for a map coordinate.

    Args:
        lat:           Latitude.
        lon:           Longitude.
        location_name: Human-readable place name (from reverse geocode or similar).

    Returns:
        Final EchoMapsState dict with keys:
          context_chunks, sources, location_id,
          interpretations, final_narrative, knowledge_graph.
    """
    initial: EchoMapsState = {
        "lat":           lat,
        "lon":           lon,
        "location_name": location_name,
    }

    log.info("Starting EchoMaps pipeline for %s (%.4f, %.4f)", location_name, lat, lon)
    result: EchoMapsState = _GRAPH.invoke(initial)
    log.info("Pipeline complete for %s", location_name)

    return dict(result)
