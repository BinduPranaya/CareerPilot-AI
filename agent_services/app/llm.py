"""
Thin wrapper around the Gemini API.

Design goal: the multi-agent graph must run end-to-end even when no
GEMINI_API_KEY is configured (e.g. during judging, CI, or local dev without
billing set up). Every agent calls `generate_json()` and supplies its own
deterministic fallback; this module only decides whether to call the real
model or skip straight to the fallback, and never lets a network/parsing
failure crash the graph.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable, Optional

logger = logging.getLogger("careerpilot.llm")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

_client_ready = False
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        _client_ready = True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Gemini client could not be initialized: %s", exc)
        _client_ready = False


def is_live() -> bool:
    """Whether a real Gemini API key is configured and the SDK loaded."""
    return _client_ready


def _extract_json(text: str) -> Optional[Any]:
    """Pull the first JSON object/array out of a model response."""
    text = text.strip()
    # strip ```json ... ``` fences if present
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # try to find the outermost {...} or [...] block
        for open_c, close_c in (("{", "}"), ("[", "]")):
            start = text.find(open_c)
            end = text.rfind(close_c)
            if start != -1 and end != -1 and end > start:
                candidate = text[start : end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
    return None


def generate_json(
    system_prompt: str,
    user_prompt: str,
    fallback: Callable[[], Any],
    *,
    agent_name: str = "agent",
) -> Any:
    """
    Ask Gemini to produce a JSON-shaped response. On any failure (no API key,
    network error, malformed JSON) the supplied `fallback` callable is used
    instead, so callers always get a usable, well-typed result.
    """
    if not _client_ready:
        return fallback()

    try:
        import google.generativeai as genai

        model = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=system_prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        response = model.generate_content(user_prompt)
        parsed = _extract_json(response.text)
        if parsed is None:
            logger.warning("[%s] Gemini returned non-JSON content; using fallback", agent_name)
            return fallback()
        return parsed
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[%s] Gemini call failed (%s); using fallback", agent_name, exc)
        return fallback()
