"""
Temporal sentiment scoring for historical eras.

Prompts the LLM to extract time periods from persona interpretations and
score each period's sentiment on a -1 (dark) → 0 (neutral) → +1 (positive) scale.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from groq import Groq

log = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a temporal analysis specialist. "
    "You read historian interpretations and identify the distinct historical periods "
    "they describe, then score the overall character of each period on a scale from "
    "-1.0 (very dark: war, mass death, oppression, catastrophe) through "
    "0.0 (neutral: transition, ambiguity, mixed events) to "
    "+1.0 (positive: prosperity, cultural flourishing, peace, founding). "
    "You output only valid JSON — no prose, no markdown fences, no commentary."
)

_USER_TEMPLATE = (
    'Four historians have interpreted the location "{location_name}".\n\n'
    "=== ARCHAEOLOGIST ===\n{archaeologist}\n\n"
    "=== POLITICAL HISTORIAN ===\n{political}\n\n"
    "=== FOLKLORIST ===\n{folklore}\n\n"
    "=== TRAUMA HISTORIAN ===\n{trauma}\n\n"
    "---\n"
    "Extract every distinct historical era or period mentioned across these four "
    "interpretations. For each period output a JSON object with these exact keys:\n"
    '  "era"          – short descriptive name (e.g. "Roman Occupation", "Golden Age")\n'
    '  "period_start" – approximate start year as an integer (negative for BCE)\n'
    '  "period_end"   – approximate end year as an integer\n'
    '  "score"        – sentiment float from -1.0 to 1.0\n'
    '  "summary"      – 1–2 sentence description of what defined this period here\n\n'
    "Return ONLY a JSON array of these objects, sorted chronologically by period_start. "
    "If no specific periods can be identified, return an empty array []."
)

# Regex to locate a JSON array anywhere in the model's response
_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)


# ── Core function ─────────────────────────────────────────────────────────────

def score_temporal_sentiment(
    interpretations: list[dict[str, Any]],
    location_name: str,
    groq_client: Groq,
) -> list[dict[str, Any]]:
    """Extract historical eras from persona interpretations and score each.

    Args:
        interpretations: List of dicts from run_all_agents; each has
                         {persona, interpretation, confidence}.
        location_name:   Human-readable place name.
        groq_client:     Authenticated Groq client instance.

    Returns:
        List of dicts, each representing one historical era:
          era           – descriptive name string
          period_start  – int (year; negative = BCE)
          period_end    – int (year)
          score         – float clamped to [-1.0, 1.0]
          summary       – 1-2 sentence description
        Sorted chronologically. Returns [] on failure.
    """
    # Index by persona for reliable slot-filling
    by_persona: dict[str, str] = {
        item["persona"]: item.get("interpretation", "").strip()
        for item in interpretations
        if item.get("interpretation")
    }
    fallback = "No interpretation available."

    user_message = _USER_TEMPLATE.format(
        location_name=location_name,
        archaeologist=by_persona.get("archaeologist", fallback),
        political=by_persona.get("political", fallback),
        folklore=by_persona.get("folklore", fallback),
        trauma=by_persona.get("trauma", fallback),
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.2,   # low temp → more reliable JSON
            max_tokens=800,
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        log.error("Groq temporal scoring error: %s", exc)
        return []

    return _parse_timeline(raw)


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_timeline(raw: str) -> list[dict[str, Any]]:
    """Extract and validate the JSON array from the model's raw response."""
    # 1. Try to parse the whole response directly
    candidates: list[str] = [raw]

    # 2. Also try any [...] substring found by regex
    m = _ARRAY_RE.search(raw)
    if m:
        candidates.append(m.group(0))

    for text in candidates:
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            continue

        if not isinstance(data, list):
            continue

        validated = [_validate_entry(entry) for entry in data if isinstance(entry, dict)]
        validated = [e for e in validated if e is not None]

        if validated:
            # Sort chronologically
            validated.sort(key=lambda e: e["period_start"])
            log.info("Parsed %d timeline entries", len(validated))
            return validated

    log.warning("Could not parse timeline JSON from model output: %r", raw[:200])
    return []


def _validate_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Coerce and validate a single timeline entry; return None if unusable."""
    try:
        era = str(entry.get("era", "Unknown Era")).strip()
        if not era:
            return None

        period_start = int(entry.get("period_start", 0))
        period_end   = int(entry.get("period_end",   period_start + 1))

        # Ensure start ≤ end
        if period_start > period_end:
            period_start, period_end = period_end, period_start

        raw_score = float(entry.get("score", 0.0))
        score = max(-1.0, min(1.0, raw_score))

        summary = str(entry.get("summary", "")).strip()

        return {
            "era":          era,
            "period_start": period_start,
            "period_end":   period_end,
            "score":        score,
            "summary":      summary,
        }
    except (TypeError, ValueError) as exc:
        log.debug("Skipping malformed timeline entry %r: %s", entry, exc)
        return None
