"""
chatbot/bot.py
--------------
Main CapitalIQ chatbot class — the orchestration layer integrating all modules.

Author  : Ali Ahmad
Project : DecodeLabs AI Project 1 — Rule-Based AI Chatbot
Batch   : 2026 | Powered by DecodeLabs

Architecture Note:
    This class contains NO domain logic. Domain knowledge lives in knowledge_base.py.
    Sanitization lives in sanitizer.py. Intent matching lives in matcher.py.
    Memory management lives in memory.py.

    bot.py's sole responsibility is pipeline orchestration:
        raw_input → sanitize → exit_check → match → personalize → respond → remember

    This design is the Separation of Concerns principle in action. Each module
    can be unit-tested, replaced, or upgraded in isolation without touching bot.py.
"""

from chatbot.sanitizer import sanitize, is_empty
from chatbot.matcher import match_intent_p1, match_intent_p2, is_exit_command
from chatbot.memory import ConversationMemory
from chatbot.knowledge_base import FALLBACK_RESPONSE, BASE_KNOWLEDGE


# ===========================================================================
# BOT METADATA
# ===========================================================================

BOT_NAME: str = "CapitalIQ"
BOT_VERSION: str = "2.0.0"
BOT_DOMAIN: str = "Finance & Investment Advisory"
AUTHOR: str = "Ali Ahmad"
BATCH: str = "2026"


# ===========================================================================
# MAIN BOT CLASS
# ===========================================================================

class CapitalIQBot:
    """
    CapitalIQ — A Rule-Based Finance & Investment Advisor Chatbot.

    Supports two operating modes (phases):
        Phase 1: Multi-keyword matching — deterministic, no confidence scoring.
                 Ideal for demonstrations of the core IPO architecture.
        Phase 2: Confidence-scored matching + conversation memory.
                 The production version — adds personalization and history.

    Attributes
    ----------
    phase : int
        Active operating phase. Must be 1 or 2.
    memory : ConversationMemory
        Session-scoped memory instance (active in both phases; used fully in P2).
    verbose : bool
        If True, prints intent name and confidence score after each response.
    """

    def __init__(self, phase: int = 2, verbose: bool = False) -> None:
        """
        Initialize the CapitalIQBot.

        Parameters
        ----------
        phase : int
            1 for Phase 1 mode; 2 for Phase 2 mode (default).
        verbose : bool
            Print intent + confidence debug info alongside responses (default: False).

        Raises
        ------
        ValueError
            If phase is not 1 or 2.
        """
        if phase not in (1, 2):
            raise ValueError(f"Invalid phase '{phase}'. Must be 1 or 2.")

        self.phase = phase
        self.verbose = verbose
        self.memory = ConversationMemory()

    # =======================================================================
    # CORE RESPONSE PIPELINE
    # =======================================================================

    def respond(self, raw_input: str) -> str:
        """
        Execute the full IPO pipeline for a single user input.

        Pipeline:
            INPUT  → sanitize raw_input
            GUARD  → return prompt if empty
            PROCESS→ route to P1 or P2 matcher
            P2 ONLY→ extract name, personalize response
            OUTPUT → record turn in memory (P2), return response string

        Parameters
        ----------
        raw_input : str
            Raw user input string, exactly as received from input().

        Returns
        -------
        str
            The bot's response to display to the user.
        """
        # ── INPUT STAGE: Sanitize ─────────────────────────────────────────
        clean = sanitize(raw_input)

        # Guard: empty or whitespace-only input must never crash the loop
        if is_empty(clean):
            return (
                "It looks like you sent an empty message. "
                "What financial topic can I help you with? "
                "Try asking about stocks, bonds, SIPs, or budgeting."
            )

        # ── PROCESS STAGE: Route to phase-appropriate matcher ─────────────
        if self.phase == 1:
            # Phase 1: simple multi-keyword matching, no confidence scoring
            response = match_intent_p1(clean)
            intent = "p1_match"
            confidence = 0.0

        else:
            # Phase 2: full confidence-scored pipeline
            intent, response, confidence = match_intent_p2(clean)

            # ── Phase 2 personalisation: name extraction ──────────────────
            extracted_name = self.memory.extract_name(clean)
            if extracted_name:
                # Store name and personalise this response
                self.memory.set_slot("user_name", extracted_name)
                response = (
                    f"Nice to meet you, {extracted_name}! "
                    f"I'm CapitalIQ, your Finance & Investment Advisor. "
                    f"{response}"
                )

            # ── Phase 2 personalisation: use stored name periodically ─────
            elif (
                self.memory.has_slot("user_name")
                and intent not in ("fallback", "none", "goodbye")
                and self.memory.turn_count > 0
                and self.memory.turn_count % 4 == 0  # every 4th turn — natural cadence
            ):
                name = self.memory.get_slot("user_name")
                response = f"{response}\n\n  💡 Great question, {name}! Keep exploring."

        # ── OUTPUT STAGE: Record in memory (both phases for history) ──────
        self.memory.add_turn(raw_input, response, intent, confidence)

        # ── Optional verbose debug output ─────────────────────────────────
        if self.verbose:
            print(f"  [DEBUG] intent='{intent}' | confidence={confidence:.3f} | phase={self.phase}")

        return response

    # =======================================================================
    # EXIT DETECTION
    # =======================================================================

    def should_exit(self, raw_input: str) -> bool:
        """
        Check if the user's raw input is an exit/quit command.

        Sanitizes the input before checking, so 'EXIT', 'Quit ', '  bye  '
        all correctly trigger a session termination.

        Parameters
        ----------
        raw_input : str
            Raw input string from input().

        Returns
        -------
        bool
            True if the session should terminate gracefully.
        """
        clean = sanitize(raw_input)
        return is_exit_command(clean)

    # =======================================================================
    # SESSION MANAGEMENT
    # =======================================================================

    def greet(self) -> str:
        """
        Return the bot's opening banner and greeting for a new session.

        Returns
        -------
        str
            Formatted multi-line welcome message string.
        """
        border = "═" * 62
        return (
            f"\n{border}\n"
            f"  💹  {BOT_NAME}  v{BOT_VERSION}  —  {BOT_DOMAIN}\n"
            f"  🎓  DecodeLabs AI Internship  |  Batch {BATCH}\n"
            f"  👤  Built by {AUTHOR}\n"
            f"{border}\n"
            f"  Ask me about: stocks · bonds · mutual funds · ETFs · crypto\n"
            f"  Also covers : inflation · compound interest · risk · SIPs\n"
            f"                budgeting · emergency funds · IPOs · portfolios\n"
            f"{border}\n"
            f"  Type 'exit' or 'quit' at any time to end the session.\n"
            f"{border}\n"
        )

    def farewell(self) -> str:
        """
        Return a farewell message and print the session history (Phase 2 only).

        Returns
        -------
        str
            Goodbye message string.
        """
        # Display session transcript only in Phase 2 (memory is fully active)
        if self.phase == 2 and self.memory.history:
            self.memory.display_history()

        summary = self.memory.get_summary()
        user_name = summary.get("user_name", "")
        name_part = (
            f", {user_name}"
            if user_name not in ("Unknown", "")
            else ""
        )

        return (
            f"\n{'═' * 62}\n"
            f"  💹  Thank you for using {BOT_NAME}{name_part}!\n"
            f"  📈  Session: {summary['total_turns']} turn(s) completed.\n"
            f"  🧠  Wealth is built slowly — through discipline and consistency.\n"
            f"  🚀  Goodbye and invest wisely!\n"
            f"{'═' * 62}\n"
        )

    # =======================================================================
    # MAIN INTERACTIVE LOOP
    # =======================================================================

    def run(self) -> None:
        """
        Launch the interactive chatbot session.

        This is the core 'while True' loop — the heartbeat of the IPO model.
        It runs indefinitely until the user issues an exit command or sends
        a keyboard interrupt (Ctrl+C).

        Loop invariant: each iteration captures exactly one user input and
        produces exactly one bot response (or terminates cleanly).
        """
        print(self.greet())

        while True:
            try:
                # ── INPUT: Capture raw user input ─────────────────────────
                raw_input = input("  You: ").strip()

            except (EOFError, KeyboardInterrupt):
                # Gracefully handle Ctrl+C and piped EOF (automated testing)
                print(f"\n\n  [{BOT_NAME}] Session interrupted. Goodbye!")
                break

            # ── EXIT CHECK: Must happen BEFORE respond() ──────────────────
            # Reason: respond() also generates a goodbye response from the
            # knowledge base, so we print it before calling farewell().
            if self.should_exit(raw_input):
                goodbye_response = self.respond(raw_input)
                print(f"\n  {BOT_NAME}: {goodbye_response}\n")
                print(self.farewell())
                break

            # ── PROCESS + OUTPUT: Generate and display response ───────────
            response = self.respond(raw_input)
            print(f"\n  {BOT_NAME}: {response}\n")

    # =======================================================================
    # SIMULATION MODE (for automated testing / Cell 10 demo)
    # =======================================================================

    def simulate(self, inputs: list[str], verbose: bool = True) -> list[dict]:
        """
        Run the bot against a list of pre-defined inputs without blocking on input().

        Used in the Cell 10 demo and unit tests to verify bot behaviour
        programmatically without requiring an interactive terminal session.

        Parameters
        ----------
        inputs : list[str]
            Ordered list of user input strings to feed the bot.
        verbose : bool
            If True, prints each exchange to stdout (default: True).

        Returns
        -------
        list[dict]
            List of result dicts with keys: input, response, intent, confidence.
        """
        results = []
        self.memory.reset()  # Start fresh for each simulation run

        if verbose:
            print(f"\n{'─' * 62}")
            print(f"  🤖  CapitalIQ Simulation Mode  |  Phase {self.phase}")
            print(f"{'─' * 62}")

        for user_input in inputs:
            # Determine intent and confidence before recording
            clean = sanitize(user_input)

            if self.should_exit(user_input):
                response = self.respond(user_input)
                intent = "goodbye"
                confidence = 1.0
            elif is_empty(clean):
                response = self.respond(user_input)
                intent = "empty_input"
                confidence = 0.0
            else:
                response = self.respond(user_input)
                # Retrieve the last recorded turn's metadata
                if self.memory.history:
                    last = self.memory.history[-1]
                    intent = last["intent"]
                    confidence = last["confidence"]
                else:
                    intent = "unknown"
                    confidence = 0.0

            result = {
                "input": user_input,
                "response": response,
                "intent": intent,
                "confidence": confidence,
            }
            results.append(result)

            if verbose:
                print(f"\n  Input      : \"{user_input}\"")
                print(f"  Intent     : {intent}  (confidence: {confidence})")
                print(f"  Response   : {response[:100]}{'...' if len(response) > 100 else ''}")
                print(f"  {'─' * 58}")

        return results


# ===========================================================================
# STANDALONE ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    # Run Phase 2 (production version) by default
    bot = CapitalIQBot(phase=2, verbose=False)
    bot.run()
