"""
chatbot/sanitizer.py
--------------------
Input sanitization and normalization pipeline for CapitalIQ.

Author  : Ali Ahmad
Project : DecodeLabs AI Project 1 — Rule-Based AI Chatbot
Batch   : 2026 | Powered by DecodeLabs

Design Rationale:
    The sanitizer is the FIRST stage of the IPO (Input-Process-Output) model.
    Raw user input is noisy: mixed case, extra whitespace, punctuation, and
    unexpected characters. Normalizing it here — before any matching logic —
    ensures the intent engine always receives a consistent, clean string.
    This prevents false negatives caused by trivial surface-form differences
    (e.g., 'Bitcoin!', 'BITCOIN', '  bitcoin  ' all become 'bitcoin').

    Keeping sanitization in its own module enforces the Single Responsibility
    Principle: if normalization rules change, only this file needs editing.
"""

import re


# ===========================================================================
# PUBLIC API
# ===========================================================================

def sanitize(raw_input: str) -> str:
    """
    Run the full sanitization pipeline on a raw user input string.

    Pipeline stages (in order):
        1. Type guard  — ensures input is a string; returns '' otherwise.
        2. Lowercase   — 'Hello' and 'hello' must be treated identically.
        3. Strip       — remove leading/trailing whitespace.
        4. Punctuation — replace non-alphanumeric chars (except apostrophes
                         and forward-slash for '50/30/20' style ratios).
        5. Collapse    — reduce multiple consecutive spaces to a single space.
        6. Final strip — catch any edge-case trailing spaces after substitution.

    Parameters
    ----------
    raw_input : str
        The raw string captured directly from the user via input().

    Returns
    -------
    str
        A clean, normalized string ready for intent matching.
        Returns an empty string '' if input is None, non-string, or blank.

    Examples
    --------
    >>> sanitize("  What's BITCOIN??  ")
    "what's bitcoin"
    >>> sanitize("   ")
    ""
    >>> sanitize(None)
    ""
    >>> sanitize("50/30/20 rule")
    "50/30/20 rule"
    """
    # Stage 1: Type guard — handle None or unexpected types gracefully
    if not isinstance(raw_input, str):
        return ""

    # Stage 2: Lowercase — normalization starts here
    cleaned = raw_input.lower()

    # Stage 3: Strip leading/trailing whitespace
    cleaned = cleaned.strip()

    # Stage 4: Remove characters that are not alphanumeric, spaces,
    #           apostrophes (contractions), or forward-slashes (ratios)
    cleaned = re.sub(r"[^a-z0-9\s'/]", " ", cleaned)

    # Stage 5: Collapse multiple consecutive spaces to a single space
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Stage 6: Final strip — the substitution may create trailing spaces
    cleaned = cleaned.strip()

    return cleaned


def is_empty(sanitized_input: str) -> bool:
    """
    Determine whether a sanitized input is effectively empty.

    Called immediately after sanitize() to short-circuit the pipeline for
    blank or whitespace-only messages before they reach the matcher.

    Parameters
    ----------
    sanitized_input : str
        A string that has already passed through sanitize().

    Returns
    -------
    bool
        True if the string is empty (length 0), False otherwise.

    Examples
    --------
    >>> is_empty("")
    True
    >>> is_empty("bitcoin")
    False
    """
    return len(sanitized_input) == 0


def tokenize(sanitized_input: str) -> list[str]:
    """
    Split a sanitized input string into individual word tokens.

    This is used by the Phase 1 and Phase 2 matchers to implement
    multi-keyword intent scoring: the input is broken into tokens,
    and each token is individually looked up in KEYWORD_INTENT_MAP.

    Parameters
    ----------
    sanitized_input : str
        A clean string that has already passed through sanitize().

    Returns
    -------
    list[str]
        A list of word tokens. Returns an empty list for empty input.

    Examples
    --------
    >>> tokenize("what is compound interest")
    ['what', 'is', 'compound', 'interest']
    >>> tokenize("")
    []
    >>> tokenize("sip")
    ['sip']
    """
    if not sanitized_input:
        return []
    # Split on whitespace — safe because sanitize() has already collapsed spaces
    return sanitized_input.split()
