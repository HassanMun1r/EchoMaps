"""
Embedding helpers using sentence-transformers (runs fully locally, no API key).
Model: all-MiniLM-L6-v2  (~80 MB, downloaded once on first use)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# Load once at import time so the model isn't reloaded on every request
_model = SentenceTransformer(EMBEDDING_MODEL)
log.info("Loaded embedding model: %s", EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return a list of embedding vectors for the given texts."""
    if not texts:
        return []
    vectors = _model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into overlapping word-count chunks.

    Uses word boundaries so chunks don't cut mid-word.
    chunk_size and overlap are measured in words (not characters).
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap  # step forward by (chunk_size - overlap)

    return chunks
