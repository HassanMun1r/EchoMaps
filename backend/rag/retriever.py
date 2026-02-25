"""
High-level retriever: orchestrates fetch → embed → upsert → search.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.rag.fetcher import get_wikidata_entities, get_wikipedia_summary
from backend.rag.vectorstore import location_exists, query_vectorstore, upsert_location_docs

log = logging.getLogger(__name__)


def _make_location_id(lat: float, lon: float) -> str:
    """Stable location key rounded to 2 decimal places (~1 km grid)."""
    return f"{round(lat, 2)}_{round(lon, 2)}"


def retrieve_context_for_location(
    lat: float,
    lon: float,
    query: str,
) -> dict[str, Any]:
    """Fetch, embed, and retrieve context for a map location.

    Pipeline:
      1. Derive a stable location_id from (lat, lon).
      2. If not already in ChromaDB: fetch Wikipedia + Wikidata, upsert.
      3. Semantic search for *query* filtered to this location.
      4. Return chunks, sources, and location_id.
    """
    location_id = _make_location_id(lat, lon)

    if not location_exists(location_id):
        log.info("Cache miss for %s — fetching remote data", location_id)
        wiki_docs = get_wikipedia_summary(lat, lon)
        wd_docs = get_wikidata_entities(lat, lon)

        all_docs = wiki_docs + [
            {
                "title": e["label"],
                "extract": f"{e['label']}: {e['description']}",
                "url": "",
                "lat": e["lat"],
                "lon": e["lon"],
            }
            for e in wd_docs
            if e.get("description")
        ]

        if all_docs:
            upsert_location_docs(all_docs, location_id)
        else:
            log.warning("No data found for location_id=%s", location_id)
            return {"chunks": [], "sources": [], "location_id": location_id}
    else:
        log.info("Cache hit for %s", location_id)

    hits = query_vectorstore(query, location_id)

    chunks = [h["text"] for h in hits]
    sources = [
        {"title": h["title"], "url": h["url"], "source": h["source"], "score": h["score"]}
        for h in hits
        if h.get("url") or h.get("title")
    ]
    # Deduplicate sources by title
    seen: set[str] = set()
    unique_sources: list[dict[str, Any]] = []
    for s in sources:
        if s["title"] not in seen:
            seen.add(s["title"])
            unique_sources.append(s)

    return {"chunks": chunks, "sources": unique_sources, "location_id": location_id}
