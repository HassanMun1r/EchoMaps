"""
Named-entity recognition over synthesised narratives.

Uses spaCy en_core_web_sm (loaded once at import time).
"""

from __future__ import annotations

import logging
from typing import Any

import spacy

log = logging.getLogger(__name__)

# Labels we care about — matches the frontend TYPE_COLORS mapping
_KEEP_LABELS: frozenset[str] = frozenset({
    "PERSON",   # people, including fictional
    "ORG",      # companies, agencies, institutions
    "GPE",      # geopolitical entities (countries, cities, states)
    "EVENT",    # named hurricanes, battles, wars, sports events
    "DATE",     # absolute or relative dates / periods
    "NORP",     # nationalities, religious or political groups
})

# Load once — relatively expensive
try:
    _NLP = spacy.load("en_core_web_sm")
    log.info("spaCy en_core_web_sm loaded")
except OSError as exc:
    _NLP = None  # type: ignore[assignment]
    log.error("Could not load spaCy model: %s — entity extraction disabled", exc)


def extract_entities(text: str) -> list[dict[str, Any]]:
    """Run NER on *text* and return entities of interest.

    Args:
        text: Arbitrary text (narrative, context chunks, etc.)

    Returns:
        List of dicts with keys:
          text   – surface form of the entity
          label  – spaCy entity label (PERSON, GPE, …)
          start  – character start offset
          end    – character end offset
    """
    if _NLP is None:
        log.warning("spaCy model unavailable — returning empty entity list")
        return []

    if not text or not text.strip():
        return []

    doc = _NLP(text)

    seen: set[str] = set()
    entities: list[dict[str, Any]] = []

    for ent in doc.ents:
        if ent.label_ not in _KEEP_LABELS:
            continue

        surface = ent.text.strip()
        if not surface or len(surface) < 2:
            continue

        # Deduplicate by (normalised text, label)
        key = (surface.lower(), ent.label_)
        if key in seen:
            continue
        seen.add(key)

        entities.append({
            "text":  surface,
            "label": ent.label_,
            "start": ent.start_char,
            "end":   ent.end_char,
        })

    return entities
