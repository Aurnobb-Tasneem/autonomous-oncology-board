"""
ml/utils/json_extract.py
========================
Robust JSON extraction from LLM text output.

LLMs frequently emit valid JSON surrounded by markdown fences, prose preambles,
or trailing notes. The standard re.search(r'\{.*\}', text, re.DOTALL) is greedy
and breaks whenever the LLM appends a citation like "see section {3}" after the
JSON block — the regex captures everything from the first '{' to the last '}',
producing a string that is not valid JSON.

This module provides a balanced-brace extractor that respects string literals,
ensuring only the first complete JSON object is extracted regardless of what
the LLM writes afterward.

Usage:
    from ml.utils.json_extract import extract_json_object

    parsed = extract_json_object(llm_response_text)
    if parsed is None:
        # trigger your fallback path
        ...
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

# Matches opening/closing markdown code fences with optional language tag.
_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    """Remove markdown ```json ... ``` fences from text."""
    return _FENCE_RE.sub("", text).strip()


def _extract_balanced(text: str) -> Optional[str]:
    """
    Walk the string from the first '{' and return the shortest balanced
    JSON-object substring that respects string literals and escape sequences.

    This correctly handles:
      - Nested objects:  {"a": {"b": 1}}
      - Strings with braces:  {"note": "see {appendix}"}
      - Escaped quotes inside strings: {"k": "say \\"hi\\""}
      - LLM prose after the JSON: {...} Note: see {3}.

    Returns None if no balanced object is found.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    i = start

    while i < len(text):
        ch = text[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if ch == "\\" and in_string:
            escape_next = True
            i += 1
            continue

        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1

    return None


def extract_json_object(text: str) -> Optional[dict]:
    """
    Robustly extract the first JSON object from arbitrary LLM output.

    Strategy ladder:
      1. Strip markdown code fences.
      2. Try json.loads on the whole stripped string (fast path).
      3. Use the balanced-brace extractor to isolate just the JSON object,
         then json.loads on that substring.
      4. Return None if all attempts fail.

    The caller is responsible for deciding what to do on None (typically
    invoking a rule-based fallback and setting from_fallback=True).

    Args:
        text: Raw LLM output string.

    Returns:
        Parsed dict, or None if no valid JSON object could be extracted.
    """
    if not text:
        return None

    cleaned = _strip_fences(text)

    # Fast path: entire stripped text is valid JSON.
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Balanced-brace extraction.
    substring = _extract_balanced(cleaned)
    if substring is None:
        # Try again on the raw text in case fences stripped something important.
        substring = _extract_balanced(text)

    if substring is not None:
        try:
            result = json.loads(substring)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError) as e:
            log.debug(f"extract_json_object: balanced extraction produced invalid JSON: {e}")

    log.debug(
        f"extract_json_object: failed to extract JSON from text "
        f"(first 120 chars): {text[:120]!r}"
    )
    return None


def extract_json_or_raise(text: str) -> dict:
    """
    Same as extract_json_object but raises ValueError on failure.

    Useful in contexts where the caller wants to catch the error explicitly
    rather than check for None.
    """
    result = extract_json_object(text)
    if result is None:
        excerpt = text[:200] if len(text) > 200 else text
        raise ValueError(
            f"Could not extract a valid JSON object from LLM output. "
            f"First 200 chars: {excerpt!r}"
        )
    return result
