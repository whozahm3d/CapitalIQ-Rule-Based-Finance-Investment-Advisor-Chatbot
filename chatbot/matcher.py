"""
chatbot/matcher.py
------------------
Intent matching and confidence scoring engine for CapitalIQ.

Author  : Ali Ahmad
Project : DecodeLabs AI Project 1 — Rule-Based AI Chatbot
Batch   : 2026 | Powered by DecodeLabs

Design Rationale:
    Phase 1 — Multi-keyword voting:
        The user's sanitized input is tokenized. Each token is looked up in
        KEYWORD_INTENT_MAP. The intent with the most token votes wins.
        This is far more robust than exact string matching — it handles
        natural phrasing like "how does compound interest work?" gracefully.

    Phase 2 — Confidence scoring:
        Raw vote counts are normalized to a [0.0, 1.0] range by dividing by
        the total number of input tokens. A configurable CONFIDENCE_THRESHOLD
        gates the output: scores below the threshold fall back to the default
        response. This mirrors production NLU classifiers (Rasa, Dialogflow)
        which also use confidence thresholds to decide when to escalate.

    Both phases check for synonym exact-match FIRST (O(1)) before performing
    token-level scoring (O(k) where k = number of tokens, typically < 20).
"""

from chatbot.knowledge_base import (
    BASE_KNOWLEDGE,
    SYNONYM_MAP,
    KEYWORD_INTENT_MAP,
    FALLBACK_RESPONSE,
    EXIT_KEYWORDS,
)
from chatbot.sanitizer import tokenize


# ===========================================================================
# CONFIGURATION CONSTANTS
# ===========================================================================

CONFIDENCE_THRESHOLD: float = 0.15  # Minimum normalized score to accept a match
# Rationale for 0.15: a single keyword in a 6-token sentence scores ~0.167,
# which is just above threshold. This allows partial matches while filtering
# pure noise inputs that happen to contain one marginally relevant word.


# ===========================================================================
# SHARED UTILITY
# ===========================================================================

def is_exit_command(sanitized_input: str) -> bool:
    """
    Check whether the user's sanitized input signals intent to end the session.

    Two-level check:
        1. Exact match against EXIT_KEYWORDS (handles single-word commands).
        2. Token-level check (handles "I want to exit now" → token 'exit').

    Parameters
    ----------
    sanitized_input : str
        Clean, normalized user input.

    Returns
    -------
    bool
        True if the session should terminate, False otherwise.

    Examples
    --------
    >>> is_exit_command("quit")
    True
    >>> is_exit_command("i want to exit now")
    True
    >>> is_exit_command("what is bitcoin")
    False
    """
    # Exact match — fastest path
    if sanitized_input in EXIT_KEYWORDS:
        return True
    # Token-level check — catches exit intent embedded in longer sentences
    tokens = tokenize(sanitized_input)
    return any(token in EXIT_KEYWORDS for token in tokens)


# ===========================================================================
# PHASE 1: MULTI-KEYWORD MATCHER
# ===========================================================================

def match_intent_p1(sanitized_input: str) -> str:
    """
    Phase 1 intent matcher using synonym resolution + multi-keyword voting.

    Algorithm:
        Step 1: Exact synonym lookup — O(1). Covers 'hi', 'btc', 'sip', etc.
        Step 2: Tokenize and tally keyword votes per intent.
        Step 3: Return the response for the intent with the most votes.
        Step 4: If no keywords match, return FALLBACK_RESPONSE.

    Why no confidence threshold in Phase 1?
        Phase 1 is designed to be a clean, minimal baseline — it matches
        or falls back, with no probabilistic gating. Phase 2 adds that layer.

    Parameters
    ----------
    sanitized_input : str
        Clean, normalized input from the sanitizer pipeline.

    Returns
    -------
    str
        The full response string from BASE_KNOWLEDGE, or FALLBACK_RESPONSE.

    Examples
    --------
    >>> match_intent_p1("bitcoin")
    '...'  # returns crypto response
    >>> match_intent_p1("xyzzy random noise")
    '...'  # returns fallback response
    """
    # Guard: empty input after sanitization
    if not sanitized_input:
        return "I didn't catch that. Could you type a question about finance or investing?"

    # Step 1: Exact synonym lookup — most common user inputs are short phrases
    if sanitized_input in SYNONYM_MAP:
        intent = SYNONYM_MAP[sanitized_input]
        return BASE_KNOWLEDGE.get(intent, FALLBACK_RESPONSE)

    # Step 2: Tokenize and vote
    tokens = tokenize(sanitized_input)
    intent_votes: dict[str, int] = {}

    for token in tokens:
        if token in KEYWORD_INTENT_MAP:
            matched_intent = KEYWORD_INTENT_MAP[token]
            # Increment vote count for this intent
            intent_votes[matched_intent] = intent_votes.get(matched_intent, 0) + 1

    # Step 3: No recognized keywords → fallback
    if not intent_votes:
        return FALLBACK_RESPONSE

    # Step 4: The intent with the most keyword votes wins
    # dict.get is used as the key function to avoid KeyError on tie resolution
    best_intent = max(intent_votes, key=lambda k: intent_votes[k])
    return BASE_KNOWLEDGE.get(best_intent, FALLBACK_RESPONSE)


# ===========================================================================
# PHASE 2: CONFIDENCE SCORING ENGINE
# ===========================================================================

def score_intents(tokens: list[str]) -> dict[str, float]:
    """
    Compute normalized confidence scores for every candidate intent.

    Scoring formula:
        raw_score  = number of tokens from the input that voted for this intent
        normalized = raw_score / total_token_count

    Normalization is critical: without it, longer sentences would artificially
    produce higher raw vote counts, skewing results toward verbose inputs.

    Parameters
    ----------
    tokens : list[str]
        Tokenized, sanitized user input.

    Returns
    -------
    dict[str, float]
        Mapping of intent_name → confidence_score in [0.0, 1.0].
        Empty dict if no tokens match any known keyword.

    Examples
    --------
    >>> score_intents(["what", "is", "bitcoin"])
    {'crypto': 0.333}
    >>> score_intents(["random", "noise", "words"])
    {}
    """
    if not tokens:
        return {}

    raw_votes: dict[str, int] = {}
    for token in tokens:
        if token in KEYWORD_INTENT_MAP:
            intent = KEYWORD_INTENT_MAP[token]
            raw_votes[intent] = raw_votes.get(intent, 0) + 1

    # Normalize by total token count to get a [0.0, 1.0] confidence score
    total_tokens = len(tokens)
    return {intent: count / total_tokens for intent, count in raw_votes.items()}


def match_intent_p2(sanitized_input: str) -> tuple[str, str, float]:
    """
    Phase 2 intent matcher with synonym resolution, confidence scoring,
    and threshold-gated fallback.

    Return contract:
        - (intent, response, confidence) on successful match
        - ("fallback", FALLBACK_RESPONSE, score) when confidence is too low
        - ("none", prompt_message, 0.0) for empty input

    Parameters
    ----------
    sanitized_input : str
        Clean, normalized input from the sanitizer pipeline.

    Returns
    -------
    tuple[str, str, float]
        3-tuple of (matched_intent_name, response_string, confidence_score).

    Examples
    --------
    >>> match_intent_p2("tell me about compound interest")
    ('compound_interest', '...full response...', 0.25)
    >>> match_intent_p2("what is the weather today")
    ('fallback', '...fallback response...', 0.0)
    """
    # Guard: empty input
    if not sanitized_input:
        return (
            "none",
            "I didn't catch that. Could you type a question about finance or investing?",
            0.0,
        )

    # Step 1: Exact synonym lookup — perfect confidence (1.0)
    if sanitized_input in SYNONYM_MAP:
        intent = SYNONYM_MAP[sanitized_input]
        response = BASE_KNOWLEDGE.get(intent, FALLBACK_RESPONSE)
        return (intent, response, 1.0)

    # Step 2: Score all intents from tokenized input
    tokens = tokenize(sanitized_input)
    scores = score_intents(tokens)

    # Step 3: No tokens matched any keyword → pure fallback
    if not scores:
        return ("fallback", FALLBACK_RESPONSE, 0.0)

    # Step 4: Pick the highest-scoring intent
    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]

    # Step 5: Threshold gate — reject weak matches
    if best_score < CONFIDENCE_THRESHOLD:
        return ("fallback", FALLBACK_RESPONSE, round(best_score, 3))

    response = BASE_KNOWLEDGE.get(best_intent, FALLBACK_RESPONSE)
    return (best_intent, response, round(best_score, 3))
