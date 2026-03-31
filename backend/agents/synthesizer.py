"""
Synthesis agent: weaves four persona interpretations into one flowing narrative.
"""

from __future__ import annotations

import logging
from typing import Any

from groq import Groq

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a master historian and writer. Your role is to receive four distinct "
    "scholarly interpretations of the same location — from an archaeologist, a "
    "political historian, a folklorist, and a trauma historian — and weave them "
    "into a single, cohesive, richly layered narrative. "
    "The narrative should flow naturally as prose, not as a list. "
    "You must explicitly credit each analytical lens using transition phrases such as "
    "'Archaeologically speaking…', 'From a political perspective…', "
    "'In local folklore and cultural memory…', and 'Through the lens of historical trauma…'. "
    "Aim for approximately 300 words. Do not repeat the source excerpts verbatim; "
    "synthesise and elevate them."
)

_USER_TEMPLATE = (
    'The location is: "{location_name}"\n\n'
    "=== ARCHAEOLOGIST ===\n{archaeologist}\n\n"
    "=== POLITICAL HISTORIAN ===\n{political}\n\n"
    "=== FOLKLORIST ===\n{folklore}\n\n"
    "=== TRAUMA HISTORIAN ===\n{trauma}\n\n"
    "---\n"
    "Now write a unified ~300-word narrative that integrates all four perspectives "
    "into flowing prose, explicitly crediting each lens."
)


def synthesis_agent(
    interpretations: list[dict[str, Any]],
    location_name: str,
    groq_client: Groq,
) -> str:
    """Synthesise four persona interpretations into one unified narrative.

    Args:
        interpretations: List of dicts from run_all_agents; each has
                         {persona, interpretation, confidence}.
        location_name:   Human-readable place name for the prompt.
        groq_client:     Authenticated Groq client instance.

    Returns:
        A ~300-word synthesised narrative string, or an error message.
    """
    # Index by persona name for reliable slot-filling
    by_persona: dict[str, str] = {}
    for item in interpretations:
        name = item.get("persona", "")
        text = item.get("interpretation", "").strip()
        if text:
            by_persona[name] = text

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
            temperature=0.65,
            max_tokens=600,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        log.error("Groq synthesis error: %s", exc)
        return f"Synthesis unavailable: {exc}"
