"""
ChromaDB vector store — local, persistent, no cloud needed.
Collection: echomaps_locations
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings

from backend.config import CHROMA_COLLECTION, CHROMA_PERSIST_DIR
from backend.rag.embedder import chunk_text, embed_texts

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialise client and collection (once at import time)
# ---------------------------------------------------------------------------

_client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIR,
    settings=Settings(anonymized_telemetry=False),
)

_collection = _client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)

log.info(
    "ChromaDB ready — collection '%s' at %s (%d docs)",
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    _collection.count(),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upsert_location_docs(docs: list[dict[str, Any]], location_id: str) -> int:
    """Chunk, embed, and upsert *docs* into ChromaDB.

    Each doc must have: title, extract, url, lat, lon  (Wikipedia)
    or:                  label, description, type, lat, lon  (Wikidata)

    Returns total number of chunks upserted.
    """
    ids: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict[str, Any]] = []
    documents: list[str] = []

    for doc_idx, doc in enumerate(docs):
        # Normalise Wikipedia vs Wikidata shapes
        title = doc.get("title") or doc.get("label") or "Unknown"
        body = doc.get("extract") or doc.get("description") or ""
        url = doc.get("url") or ""
        source = "wikipedia" if "extract" in doc else "wikidata"

        if not body.strip():
            continue

        chunks = chunk_text(body)
        chunk_embeddings = embed_texts(chunks)

        for chunk_idx, (chunk, emb) in enumerate(zip(chunks, chunk_embeddings)):
            chunk_id = f"{location_id}_{doc_idx}_{chunk_idx}"
            ids.append(chunk_id)
            embeddings.append(emb)
            documents.append(chunk)
            metadatas.append(
                {
                    "source": source,
                    "title": title,
                    "url": url,
                    "lat": float(doc.get("lat", 0)),
                    "lon": float(doc.get("lon", 0)),
                    "location_id": location_id,
                }
            )

    if not ids:
        log.info("upsert_location_docs: nothing to upsert for location_id=%s", location_id)
        return 0

    _collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    log.info("Upserted %d chunks for location_id=%s", len(ids), location_id)
    return len(ids)


def query_vectorstore(
    query_text: str,
    location_id: str,
    n_results: int = 8,
) -> list[dict[str, Any]]:
    """Semantic search filtered to a specific location.

    Returns list of {text, source, title, url, lat, lon}.
    """
    query_emb = embed_texts([query_text])[0]

    results = _collection.query(
        query_embeddings=[query_emb],
        n_results=n_results,
        where={"location_id": location_id},
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict[str, Any]] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, dist in zip(docs, metas, distances):
        hits.append(
            {
                "text": text,
                "source": meta.get("source", ""),
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                "lat": meta.get("lat", 0.0),
                "lon": meta.get("lon", 0.0),
                "score": round(1 - dist, 4),  # cosine similarity
            }
        )

    return hits


def location_exists(location_id: str) -> bool:
    """Return True if we already have chunks stored for this location."""
    results = _collection.get(
        where={"location_id": location_id},
        limit=1,
        include=[],
    )
    return len(results.get("ids", [])) > 0
