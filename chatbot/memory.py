"""
chatbot/memory.py
-----------------
Conversation memory and session history module for CapitalIQ.

Author  : Ali Ahmad
Project : DecodeLabs AI Project 1 — Rule-Based AI Chatbot
Batch   : 2026 | Powered by DecodeLabs

Design Rationale:
    A stateless chatbot treats every message in isolation — it has no awareness
    of what the user said two turns ago. This module adds a lightweight,
    session-scoped memory layer with two capabilities:

        1. History Log — an ordered list of every turn (user message + bot reply
           + metadata). Displayed as a transcript at the end of the session.

        2. Slot Store — a key-value dictionary for named entities extracted
           during the conversation (e.g., user's name, last topic discussed).
           In production NLP pipelines (Rasa, Dialogflow, LangChain), this
           pattern is called 'slot filling' or 'context tracking'.

    The memory module is intentionally decoupled from the matcher — it neither
    reads intent data nor writes responses. It is a pure data store that the
    bot orchestration layer (bot.py) populates and queries.
"""

import re
from datetime import datetime


class ConversationMemory:
    """
    Session-scoped memory store for a single CapitalIQ chat session.

    Attributes
    ----------
    history : list[dict]
        Ordered list of turn dictionaries. Each dict contains:
            turn        (int)   — 1-indexed turn number
            timestamp   (str)   — HH:MM:SS at time of turn
            user        (str)   — raw user message as typed
            bot         (str)   — bot's full response string
            intent      (str)   — matched intent or 'fallback'
            confidence  (float) — Phase 2 confidence score (0.0–1.0)
    slots : dict[str, str]
        Named slot store. Persists across turns within the session.
    turn_count : int
        Total number of completed conversational turns.
    """

    def __init__(self) -> None:
        """Initialize a fresh, empty memory store for a new session."""
        self.history: list[dict] = []         # Ordered conversation transcript
        self.slots: dict[str, str] = {}       # Named entity / context slots
        self.turn_count: int = 0              # Completed turn counter

    # =======================================================================
    # HISTORY MANAGEMENT
    # =======================================================================

    def add_turn(
        self,
        user_input: str,
        bot_response: str,
        intent: str = "unknown",
        confidence: float = 0.0,
    ) -> None:
        """
        Record one complete conversational turn (user message + bot reply).

        This method is called by bot.py at the end of every successful turn.
        It also auto-updates the 'last_intent' slot for context continuity.

        Parameters
        ----------
        user_input : str
            The raw (unsanitized) user message exactly as typed.
        bot_response : str
            The bot's response string.
        intent : str
            The matched intent name from the matcher (default: 'unknown').
        confidence : float
            The Phase 2 confidence score, 0.0–1.0 (default: 0.0 for Phase 1).
        """
        self.turn_count += 1

        turn_record = {
            "turn": self.turn_count,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "user": user_input,
            "bot": bot_response,
            "intent": intent,
            "confidence": round(confidence, 3),
        }
        self.history.append(turn_record)

        # Auto-update the last_intent slot — used for context-aware follow-ups
        self.set_slot("last_intent", intent)

    def get_history(self) -> list[dict]:
        """
        Return the full conversation history as a list of turn dictionaries.

        Returns
        -------
        list[dict]
            All recorded turns in chronological order. Empty list if no turns.
        """
        return self.history

    def display_history(self) -> None:
        """
        Print the full session transcript to stdout in formatted, readable style.

        Called at the end of a session (from bot.farewell()) to give the user
        a complete summary of what was discussed.
        """
        if not self.history:
            print("\n  No conversation history to display.")
            return

        separator = "=" * 62

        print(f"\n{separator}")
        print("  📋  SESSION TRANSCRIPT  —  CapitalIQ")
        print(f"{separator}")

        for turn in self.history:
            # Truncate bot response for display (full text is stored in history)
            bot_preview = (
                turn["bot"][:75] + "..." if len(turn["bot"]) > 75 else turn["bot"]
            )
            print(f"\n  ┌─ Turn {turn['turn']} @ {turn['timestamp']}")
            print(f"  │  🧑 You : {turn['user']}")
            print(f"  │  🤖 Bot : {bot_preview}")
            print(f"  └─ 📊 Intent: {turn['intent']}  |  Confidence: {turn['confidence']}")

        print(f"\n{separator}")
        print(f"  Session complete. Total turns: {self.turn_count}")
        print(f"{separator}\n")

    # =======================================================================
    # SLOT MANAGEMENT
    # =======================================================================

    def set_slot(self, key: str, value: str) -> None:
        """
        Store a named entity or context variable in the slot store.

        Parameters
        ----------
        key : str
            Slot name, e.g., 'user_name', 'last_intent'.
        value : str
            Value to associate with this slot key.
        """
        self.slots[key] = str(value)  # Coerce to str for type safety

    def get_slot(self, key: str, default: str = "") -> str:
        """
        Retrieve a named slot value from the store.

        Parameters
        ----------
        key : str
            Slot name to look up.
        default : str
            Returned if the slot has not been set (default: empty string).

        Returns
        -------
        str
            The stored slot value, or default if the slot is absent.
        """
        return self.slots.get(key, default)

    def has_slot(self, key: str) -> bool:
        """
        Check whether a named slot exists and contains a non-empty value.

        Parameters
        ----------
        key : str
            Slot name to check.

        Returns
        -------
        bool
            True if the slot exists and its value is non-empty, False otherwise.
        """
        return key in self.slots and bool(self.slots[key])

    # =======================================================================
    # NAME EXTRACTION
    # =======================================================================

    def extract_name(self, sanitized_input: str) -> str | None:
        """
        Attempt to extract the user's name from common introductory patterns.

        Recognized patterns:
            - "my name is <Name>"
            - "i am <Name>"
            - "i'm <Name>"
            - "call me <Name>"

        Parameters
        ----------
        sanitized_input : str
            Clean, lowercased input from the sanitizer pipeline.

        Returns
        -------
        str | None
            The extracted name in Title Case if a pattern matched, None otherwise.

        Examples
        --------
        >>> mem = ConversationMemory()
        >>> mem.extract_name("my name is ali")
        'Ali'
        >>> mem.extract_name("what is bitcoin")
        None
        """
        patterns = [
            r"my name is (\w+)",
            r"\bi am (\w+)",
            r"\bi'm (\w+)",
            r"call me (\w+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, sanitized_input)
            if match:
                # Title-case to make the name look natural in responses
                return match.group(1).title()
        return None

    # =======================================================================
    # SESSION UTILITIES
    # =======================================================================

    def get_summary(self) -> dict:
        """
        Return a summary dictionary of the session's key statistics.

        Returns
        -------
        dict
            Keys: total_turns, unique_intents, user_name (if known),
                  most_discussed (most frequently matched intent).
        """
        if not self.history:
            return {"total_turns": 0, "unique_intents": [], "most_discussed": None}

        # Count intent frequencies (excluding fallbacks and system intents)
        intent_counts: dict[str, int] = {}
        for turn in self.history:
            intent = turn["intent"]
            if intent not in ("fallback", "none", "unknown", "goodbye"):
                intent_counts[intent] = intent_counts.get(intent, 0) + 1

        most_discussed = (
            max(intent_counts, key=lambda k: intent_counts[k])
            if intent_counts
            else None
        )

        return {
            "total_turns": self.turn_count,
            "unique_intents": list(set(
                t["intent"] for t in self.history
                if t["intent"] not in ("fallback", "none", "unknown")
            )),
            "user_name": self.get_slot("user_name") or "Unknown",
            "most_discussed": most_discussed,
        }

    def reset(self) -> None:
        """
        Clear all history and slots to start a fresh session.

        Useful for unit testing or restarting the bot without
        re-instantiating the object.
        """
        self.history = []
        self.slots = {}
        self.turn_count = 0
