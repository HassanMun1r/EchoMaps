"""
Historian agent runner.

Exposes:
  run_persona_agent(persona_name, context_chunks, location_name, groq_client) -> dict
  run_all_agents(context_chunks, location_name, groq_client) -> list[dict]
"""

from __future__ import annotations

import logging
import re
from typing import Any

from groq import Groq

from backend.agents.personas import PERSONAS

log = logging.getLogger(__name__)

_INTERPRETATION_PROMPT = (
    "Here are historical and cultural source excerpts about the location "
    '"{location_name}":\n\n'
    "{context}\n\n"
    "---\n"
    "Using only the information above, write a focused ~150-word interpretation "
    "of this location from your designated perspective. "
    "{focus_hint}\n\n"
    "End your response with a single line in this exact format (no extra text):\n"
    "CONFIDENCE: <number between 0.0 and 1.0>"
)

_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def run_persona_agent(
    persona_name: str,
    context_chunks: list[str],
    location_name: str,
    groq_client: Groq,
) -> dict[str, Any]:
    """Run a single historian persona agent against the provided context.

    Returns a dict with keys:
      persona        – persona identifier
      interpretation – ~150-word analysis from the persona's viewpoint
      confidence     – float 0-1 self-assessed by the model
    """
    persona = PERSONAS.get(persona_name)
    if persona is None:
        raise ValueError(f"Unknown persona: {persona_name!r}. "
                         f"Valid names: {list(PERSONAS)}")

    if not context_chunks:
        return {
            "persona": persona_name,
            "interpretation": "Insufficient source material to generate an interpretation.",
            "confidence": 0.0,
        }

    # Trim context to avoid exceeding context window (≈3 000 chars)
    context_text = "\n\n---\n\n".join(context_chunks)
    if len(context_text) > 3_000:
        context_text = context_text[:3_000] + "\n[… truncated …]"

    user_message = _INTERPRETATION_PROMPT.format(
        location_name=location_name,
        context=context_text,
        focus_hint=persona["focus_hint"],
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": persona["system"]},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.55,
            max_tokens=320,
        )
    except Exception as exc:
        log.error("Groq API error for persona %s: %s", persona_name, exc)
        return {
            "persona": persona_name,
            "interpretation": f"Agent unavailable: {exc}",
            "confidence": 0.0,
        }

    raw = response.choices[0].message.content or ""

    # Extract and strip the confidence line
    confidence = 0.5  # default if model omits it
    match = _CONFIDENCE_RE.search(raw)
    if match:
        try:
            confidence = float(match.group(1))
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            pass
        # Remove the CONFIDENCE line from the displayed text
        raw = _CONFIDENCE_RE.sub("", raw).strip()

    return {
        "persona": persona_name,
        "interpretation": raw.strip(),
        "confidence": confidence,
    }


def run_all_agents(
    context_chunks: list[str],
    location_name: str,
    groq_client: Groq,
) -> list[dict[str, Any]]:
    """Run all four historian persona agents sequentially.

    Returns a list of four result dicts (same structure as run_persona_agent),
    one per persona, in definition order.
    """
    results = []
    for persona_name in PERSONAS:
        log.info("Running historian agent: %s", persona_name)
        result = run_persona_agent(
            persona_name=persona_name,
            context_chunks=context_chunks,
            location_name=location_name,
            groq_client=groq_client,
        )
        results.append(result)
    return results
