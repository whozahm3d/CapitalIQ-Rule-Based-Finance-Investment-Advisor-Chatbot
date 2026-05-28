"""
tests/test_chatbot.py
---------------------
Unit test suite for the CapitalIQ chatbot modules.

Author  : Ali Ahmad
Project : DecodeLabs AI Project 1 — Rule-Based AI Chatbot
Batch   : 2026 | Powered by DecodeLabs

Coverage:
    - sanitizer.py   : sanitize(), is_empty(), tokenize()
    - matcher.py     : match_intent_p1(), match_intent_p2(), score_intents(),
                       is_exit_command()
    - memory.py      : add_turn(), get_slot(), set_slot(), extract_name(),
                       display_history(), get_summary(), reset()
    - bot.py         : CapitalIQBot.respond(), should_exit(), simulate()

Run with:
    python -m pytest tests/test_chatbot.py -v
    # or from the project root:
    python -m pytest -v
"""

import sys
import os

# Ensure the project root is on sys.path so imports work when running tests
# directly (without installing the package).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest

from chatbot.sanitizer import sanitize, is_empty, tokenize
from chatbot.matcher import (
    match_intent_p1,
    match_intent_p2,
    score_intents,
    is_exit_command,
    CONFIDENCE_THRESHOLD,
)
from chatbot.memory import ConversationMemory
from chatbot.bot import CapitalIQBot
from chatbot.knowledge_base import BASE_KNOWLEDGE, FALLBACK_RESPONSE


# ===========================================================================
# TEST CLASS 1: SANITIZER
# ===========================================================================

class TestSanitizer(unittest.TestCase):
    """Tests for chatbot/sanitizer.py"""

    # ── sanitize() ──────────────────────────────────────────────────────

    def test_sanitize_lowercase(self):
        """Mixed-case input should be fully lowercased."""
        self.assertEqual(sanitize("BITCOIN"), "bitcoin")

    def test_sanitize_strip_whitespace(self):
        """Leading and trailing whitespace must be stripped."""
        self.assertEqual(sanitize("  hello  "), "hello")

    def test_sanitize_internal_whitespace_collapsed(self):
        """Multiple internal spaces should collapse to one."""
        self.assertEqual(sanitize("what   is   compound   interest"), "what is compound interest")

    def test_sanitize_removes_special_characters(self):
        """Special characters (except apostrophes/slashes) must be removed."""
        self.assertEqual(sanitize("bitcoin!!!"), "bitcoin")
        self.assertEqual(sanitize("what??"), "what")

    def test_sanitize_preserves_apostrophes(self):
        """Apostrophes in contractions must be preserved."""
        result = sanitize("what's a bond?")
        self.assertIn("what's", result)

    def test_sanitize_preserves_slash_for_ratio(self):
        """Forward slash should be preserved for '50/30/20' style input."""
        result = sanitize("50/30/20 rule")
        self.assertIn("50/30/20", result)

    def test_sanitize_empty_string(self):
        """Empty string should return empty string."""
        self.assertEqual(sanitize(""), "")

    def test_sanitize_whitespace_only(self):
        """Whitespace-only input should return empty string."""
        self.assertEqual(sanitize("     "), "")

    def test_sanitize_none_returns_empty(self):
        """None input must not raise an exception; should return ''."""
        self.assertEqual(sanitize(None), "")

    def test_sanitize_non_string_returns_empty(self):
        """Non-string input (int, list) must return '' without crashing."""
        self.assertEqual(sanitize(12345), "")
        self.assertEqual(sanitize(["hello"]), "")

    # ── is_empty() ──────────────────────────────────────────────────────

    def test_is_empty_true_for_empty_string(self):
        self.assertTrue(is_empty(""))

    def test_is_empty_false_for_non_empty(self):
        self.assertFalse(is_empty("bitcoin"))

    # ── tokenize() ──────────────────────────────────────────────────────

    def test_tokenize_splits_correctly(self):
        """Standard sentence should split into individual word tokens."""
        self.assertEqual(tokenize("what is compound interest"), ["what", "is", "compound", "interest"])

    def test_tokenize_single_word(self):
        """Single word input should return a one-element list."""
        self.assertEqual(tokenize("sip"), ["sip"])

    def test_tokenize_empty_returns_empty_list(self):
        """Empty string should return an empty list."""
        self.assertEqual(tokenize(""), [])


# ===========================================================================
# TEST CLASS 2: MATCHER
# ===========================================================================

class TestMatcher(unittest.TestCase):
    """Tests for chatbot/matcher.py"""

    # ── is_exit_command() ───────────────────────────────────────────────

    def test_exit_exact_match(self):
        """Single exact exit keywords must be detected."""
        for word in ["exit", "quit", "bye", "goodbye", "done"]:
            with self.subTest(word=word):
                self.assertTrue(is_exit_command(word))

    def test_exit_embedded_in_sentence(self):
        """Exit keyword embedded in a longer sentence must still trigger."""
        self.assertTrue(is_exit_command("i want to exit now"))

    def test_exit_false_for_non_exit(self):
        """Regular questions must not trigger exit."""
        self.assertFalse(is_exit_command("what is a stock"))
        self.assertFalse(is_exit_command("tell me about bitcoin"))

    # ── match_intent_p1() ───────────────────────────────────────────────

    def test_p1_exact_synonym_greeting(self):
        """'hello' should resolve to greeting response."""
        response = match_intent_p1("hello")
        self.assertEqual(response, BASE_KNOWLEDGE["greeting"])

    def test_p1_exact_synonym_crypto(self):
        """'btc' should resolve to crypto response."""
        response = match_intent_p1("btc")
        self.assertEqual(response, BASE_KNOWLEDGE["crypto"])

    def test_p1_keyword_matching_compound_interest(self):
        """Multi-word natural sentence with 'compound interest' keywords."""
        response = match_intent_p1("tell me about compound interest")
        self.assertEqual(response, BASE_KNOWLEDGE["compound_interest"])

    def test_p1_keyword_matching_stocks(self):
        """Sentence containing 'stock' should return stocks response."""
        response = match_intent_p1("how do stocks work")
        self.assertEqual(response, BASE_KNOWLEDGE["stocks"])

    def test_p1_fallback_for_unknown(self):
        """Completely unrelated input should return FALLBACK_RESPONSE."""
        response = match_intent_p1("purple dinosaurs playing chess")
        self.assertEqual(response, FALLBACK_RESPONSE)

    def test_p1_empty_input_does_not_crash(self):
        """Empty sanitized input must return a safe prompt, not crash."""
        response = match_intent_p1("")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    # ── match_intent_p2() ───────────────────────────────────────────────

    def test_p2_returns_three_tuple(self):
        """Phase 2 matcher must always return a 3-tuple."""
        result = match_intent_p2("bitcoin")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_p2_exact_match_confidence_is_1(self):
        """Exact synonym match must return confidence of 1.0."""
        intent, response, confidence = match_intent_p2("sip")
        self.assertEqual(confidence, 1.0)
        self.assertEqual(intent, "sip")

    def test_p2_fallback_for_noise(self):
        """Noise input should return intent='fallback'."""
        intent, response, confidence = match_intent_p2("purple unicorn dance party")
        self.assertEqual(intent, "fallback")
        self.assertEqual(response, FALLBACK_RESPONSE)

    def test_p2_confidence_within_range(self):
        """Confidence score must always be in [0.0, 1.0]."""
        _, _, confidence = match_intent_p2("what is a mutual fund")
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_p2_empty_input_returns_none_intent(self):
        """Empty input must return intent='none' and not crash."""
        intent, response, confidence = match_intent_p2("")
        self.assertEqual(intent, "none")
        self.assertIsInstance(response, str)

    # ── score_intents() ─────────────────────────────────────────────────

    def test_score_intents_returns_dict(self):
        """score_intents must return a dict."""
        result = score_intents(["bitcoin", "blockchain"])
        self.assertIsInstance(result, dict)

    def test_score_intents_normalized(self):
        """All scores must be in (0.0, 1.0]."""
        scores = score_intents(["bitcoin", "blockchain", "defi"])
        for score in scores.values():
            self.assertGreater(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_score_intents_empty_tokens(self):
        """Empty token list must return empty dict without crashing."""
        result = score_intents([])
        self.assertEqual(result, {})


# ===========================================================================
# TEST CLASS 3: MEMORY
# ===========================================================================

class TestConversationMemory(unittest.TestCase):
    """Tests for chatbot/memory.py"""

    def setUp(self):
        """Create a fresh ConversationMemory instance before each test."""
        self.mem = ConversationMemory()

    def test_initial_state_empty(self):
        """Freshly initialized memory must have empty history and slots."""
        self.assertEqual(self.mem.history, [])
        self.assertEqual(self.mem.slots, {})
        self.assertEqual(self.mem.turn_count, 0)

    def test_add_turn_increments_count(self):
        """add_turn() must increment turn_count."""
        self.mem.add_turn("hello", "Hi!", "greeting", 1.0)
        self.assertEqual(self.mem.turn_count, 1)

    def test_add_turn_stores_data_correctly(self):
        """add_turn() must store all provided data in history."""
        self.mem.add_turn("what is a stock", "Stocks are...", "stocks", 0.5)
        turn = self.mem.history[0]
        self.assertEqual(turn["user"], "what is a stock")
        self.assertEqual(turn["intent"], "stocks")
        self.assertAlmostEqual(turn["confidence"], 0.5, places=2)

    def test_add_turn_updates_last_intent_slot(self):
        """add_turn() must auto-update the 'last_intent' slot."""
        self.mem.add_turn("bitcoin?", "Crypto info...", "crypto", 0.8)
        self.assertEqual(self.mem.get_slot("last_intent"), "crypto")

    def test_set_and_get_slot(self):
        """set_slot() and get_slot() must work correctly."""
        self.mem.set_slot("user_name", "Ali")
        self.assertEqual(self.mem.get_slot("user_name"), "Ali")

    def test_get_slot_default(self):
        """get_slot() on an absent key must return the default value."""
        self.assertEqual(self.mem.get_slot("nonexistent", "default_val"), "default_val")

    def test_has_slot_true(self):
        """has_slot() must return True when slot is set and non-empty."""
        self.mem.set_slot("user_name", "Ali")
        self.assertTrue(self.mem.has_slot("user_name"))

    def test_has_slot_false(self):
        """has_slot() must return False for absent slots."""
        self.assertFalse(self.mem.has_slot("user_name"))

    def test_extract_name_my_name_is(self):
        """extract_name() must detect 'my name is <Name>' pattern."""
        result = self.mem.extract_name("my name is ali")
        self.assertEqual(result, "Ali")

    def test_extract_name_i_am(self):
        """extract_name() must detect 'i am <Name>' pattern."""
        result = self.mem.extract_name("i am ahmad")
        self.assertEqual(result, "Ahmad")

    def test_extract_name_call_me(self):
        """extract_name() must detect 'call me <Name>' pattern."""
        result = self.mem.extract_name("call me ali")
        self.assertEqual(result, "Ali")

    def test_extract_name_none_for_regular_input(self):
        """extract_name() must return None if no name pattern is found."""
        result = self.mem.extract_name("what is bitcoin")
        self.assertIsNone(result)

    def test_reset_clears_all_state(self):
        """reset() must clear history, slots, and turn_count."""
        self.mem.add_turn("hi", "Hello!", "greeting", 1.0)
        self.mem.set_slot("user_name", "Ali")
        self.mem.reset()
        self.assertEqual(self.mem.history, [])
        self.assertEqual(self.mem.slots, {})
        self.assertEqual(self.mem.turn_count, 0)

    def test_get_history_returns_list(self):
        """get_history() must return a list."""
        self.assertIsInstance(self.mem.get_history(), list)

    def test_get_summary_structure(self):
        """get_summary() must return a dict with the expected keys."""
        self.mem.add_turn("hi", "Hello!", "greeting", 1.0)
        summary = self.mem.get_summary()
        self.assertIn("total_turns", summary)
        self.assertIn("unique_intents", summary)
        self.assertIn("most_discussed", summary)


# ===========================================================================
# TEST CLASS 4: BOT (Integration)
# ===========================================================================

class TestCapitalIQBot(unittest.TestCase):
    """Integration tests for chatbot/bot.py"""

    def setUp(self):
        """Create Phase 2 bot instance before each test."""
        self.bot = CapitalIQBot(phase=2, verbose=False)

    def test_invalid_phase_raises(self):
        """Passing an invalid phase must raise ValueError."""
        with self.assertRaises(ValueError):
            CapitalIQBot(phase=3)

    def test_respond_returns_string(self):
        """respond() must always return a string."""
        result = self.bot.respond("what is bitcoin")
        self.assertIsInstance(result, str)

    def test_respond_empty_input_no_crash(self):
        """Empty input must return a safe prompt string, not raise."""
        result = self.bot.respond("")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_respond_whitespace_only_no_crash(self):
        """Whitespace-only input must not crash the bot."""
        result = self.bot.respond("     ")
        self.assertIsInstance(result, str)

    def test_respond_special_chars_no_crash(self):
        """Input with only special characters must not crash the bot."""
        result = self.bot.respond("!@#$%^&*()")
        self.assertIsInstance(result, str)

    def test_should_exit_true_for_quit(self):
        """should_exit() must return True for 'quit'."""
        self.assertTrue(self.bot.should_exit("quit"))

    def test_should_exit_false_for_question(self):
        """should_exit() must return False for a regular question."""
        self.assertFalse(self.bot.should_exit("what is inflation"))

    def test_greet_returns_string(self):
        """greet() must return a non-empty string."""
        result = self.bot.greet()
        self.assertIsInstance(result, str)
        self.assertIn("CapitalIQ", result)

    def test_farewell_returns_string(self):
        """farewell() must return a non-empty string."""
        result = self.bot.farewell()
        self.assertIsInstance(result, str)

    def test_simulate_returns_list(self):
        """simulate() must return a list of result dicts."""
        inputs = ["hello", "what is a stock", "quit"]
        results = self.bot.simulate(inputs, verbose=False)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)

    def test_simulate_result_structure(self):
        """Each result dict from simulate() must have required keys."""
        results = self.bot.simulate(["bitcoin"], verbose=False)
        self.assertIn("input", results[0])
        self.assertIn("response", results[0])
        self.assertIn("intent", results[0])
        self.assertIn("confidence", results[0])

    def test_phase1_bot_responds(self):
        """Phase 1 bot must also respond correctly to known inputs."""
        bot_p1 = CapitalIQBot(phase=1, verbose=False)
        result = bot_p1.respond("bitcoin")
        self.assertEqual(result, BASE_KNOWLEDGE["crypto"])

    def test_memory_records_turns(self):
        """After calling respond(), memory must have recorded the turn."""
        self.bot.respond("what is inflation")
        self.assertEqual(self.bot.memory.turn_count, 1)

    def test_name_extraction_personalises_response(self):
        """Introducing a name should trigger personalised response."""
        response = self.bot.respond("my name is Ali")
        self.assertIn("Ali", response)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    # Run with verbose output when executed directly
    unittest.main(verbosity=2)
